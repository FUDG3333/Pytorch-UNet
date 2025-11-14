import argparse
import logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from unet import UNet
from utils.data_loading import CarvanaDataset
from utils.dice_score import dice_loss
from evaluate import evaluate  # 你的 evaluate 函数


def train_model(
        model,
        device,
        dataset,
        epochs: int = 10,
        batch_size: int = 1,
        learning_rate: float = 1e-6,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-5,
        momentum: float = 0.9,
        gradient_clipping: float = 1.0,
        load_checkpoint: str = None,  # 添加这个参数
):
    # Split into train / validation
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # Wandb logging
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # Optimizer, scheduler, criterion
    optimizer = optim.AdamW(model.parameters(),
                          lr=learning_rate, 
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=3
    )
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 类别权重
    weights = torch.tensor([
        0.03, 3.2, 7.6, 2.7, 4.1, 3.4, 1.3, 1.6, 0.9, 2.3,
        2.3, 1.8, 1.2, 2.3, 2.0, 0.4, 4.0, 2.8, 1.6, 1.4, 2.8
    ], dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=255)

    global_step = 0
    best_val_score = 0
    start_epoch = 1

    # 加载 checkpoint（如果指定了）
    if load_checkpoint:
        try:
            checkpoint = torch.load(load_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = checkpoint["epoch"] + 1
            logging.info(f'✅ Checkpoint loaded from {load_checkpoint}')
            logging.info(f'   Resuming from epoch {start_epoch}')
        except Exception as e:
            logging.error(f'Failed to load checkpoint: {e}')
            logging.warning('Training from scratch instead')

    for epoch in range(start_epoch, epochs + 1):

        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({'train loss': loss.item(), 'step': global_step, 'epoch': epoch})
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation
                division_step = (n_train // (5 * batch_size))
                if division_step > 0 and global_step % division_step == 0:
                    val_score = evaluate(model, val_loader, device, amp)
                    scheduler.step(val_score)
                    logging.info(f'Validation Dice score: {val_score}')

        # 在每个 epoch 结束时评估
        val_score = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score)
        logging.info(f'Validation Dice score: {val_score}')
        
        # 保存最佳模型
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), './checkpoints/best_model.pth')
            logging.info(f'New best model saved! (Validation Dice: {val_score:.4f})')

        # if save_checkpoint:
        #     Path('./checkpoints/').mkdir(parents=True, exist_ok=True)
        #     state_dict = model.state_dict()
        #     # state_dict['mask_values'] = dataset.mask_values
        #     torch.save(state_dict, f'./checkpoints/checkpoint_epoch{epoch}.pth')
        #     logging.info(f'Checkpoint {epoch} saved!')
        if save_checkpoint:
            Path('./checkpoints/').mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(checkpoint, f'./checkpoints/checkpoint_epoch{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved!')



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--batch-size', '-b', dest='batch_size', type=int, default=1)
    parser.add_argument('--learning-rate', '-l', dest='lr', type=float, default=1e-5)
    parser.add_argument('--load', '-f', type=str, default=False)
    parser.add_argument('--scale', '-s', type=float, default=0.5)
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--bilinear', action='store_true', default=False)
    parser.add_argument('--classes', '-c', type=int, default=21)
    parser.add_argument('--weight-decay', type=float, default=1e-5, 
                      help='Weight decay for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum for optimizer')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # -------------------------------
    # 初始化 dataset
    dir_img = Path("data/VOCdevkit/VOC2012/JPEGImages")
    dir_mask = Path("data/VOCdevkit/VOC2012/SegmentationClass")





    img_ids = [p.stem for p in dir_img.glob("*.jpg") if (dir_mask / (p.stem + ".png")).exists()]
    print("Images with masks:", len(img_ids))

    dataset = CarvanaDataset(images_dir=dir_img, mask_dir=dir_mask, scale=args.scale,ids=img_ids)
    # dataset.ids = img_ids

    # 初始化模型
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(device=device, memory_format=torch.channels_last)
    logging.info(f'Network:\n\t{model.n_channels} input channels\n\t{model.n_classes} output channels (classes)\n\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    # 删除这一段（旧的加载逻辑）
    # start_epoch = 1
    # if args.load:
    #     checkpoint = torch.load(args.load, map_location=device)
    #     ...

    # 开始训练
    try:
        train_model(
            model=model,
            device=device,
            dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.val / 100,
            img_scale=args.scale,
            amp=args.amp,
            load_checkpoint=args.load  # 传递 checkpoint 路径
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! Enabling checkpointing...')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            device=device,
            dataset=dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            val_percent=args.val / 100,
            img_scale=args.scale,
            amp=args.amp,
            load_checkpoint=args.load  # 传递 checkpoint 路径
        )
