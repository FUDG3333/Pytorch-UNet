import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


# -----------------------------
# VOC 调色板定义
# -----------------------------
def voc_colormap(N=256):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    return cmap


# -----------------------------
# 模型预测
# -----------------------------
def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    # 添加标准化
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = normalize(img)  # 添加标准化
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        
        if net.n_classes > 1:
            # 进一步降低温度系数
            temperature = 0.5
            output = output / temperature
            output = F.softmax(output, dim=1)
            
            # 打印类别概率
            probs = output.mean(dim=(2,3))
            max_prob, pred_class = probs.max(dim=1)
            print(f"Predicted dominant class: {pred_class.item()} with probability: {max_prob.item():.4f}")
            
            # 使用更激进的阈值策略
            mask = output.argmax(dim=1)
            confidence = output.max(dim=1)[0]
            
            # 只保留高置信度预测
            high_conf = confidence > 0.2
            mask[~high_conf] = 0
            
            # 打印置信度统计
            print(f"Confidence stats - Mean: {confidence.mean():.4f}, Max: {confidence.max():.4f}")
            
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


# -----------------------------
# 参数解析
# -----------------------------
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames or folder of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', help='Output folder (default: predictions/)')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize results after prediction')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Threshold for mask (binary only)')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=21, help='Number of classes')
    return parser.parse_args()

import torch
import matplotlib.pyplot as plt
import logging
from unet import UNet

def analyze_checkpoint(checkpoint_path):
    logging.basicConfig(level=logging.INFO)
    
    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 基本信息
    logging.info(f"Checkpoint keys: {checkpoint.keys()}")
    
    # 检查权重分布
    for name, weight in checkpoint.items():
        if isinstance(weight, torch.Tensor):
            logging.info(f"\n{name}:")
            logging.info(f"Shape: {weight.shape}")
            logging.info(f"Mean: {weight.mean():.4f}")
            logging.info(f"Std: {weight.std():.4f}")
            logging.info(f"Min: {weight.min():.4f}")
            logging.info(f"Max: {weight.max():.4f}")
            
            # 绘制权重分布直方图
            plt.figure(figsize=(10, 4))
            plt.hist(weight.numpy().flatten(), bins=50)
            plt.title(f'Weight Distribution - {name}')
            plt.xlabel('Weight Value')
            plt.ylabel('Count')
            plt.savefig(f'weight_dist_{name.replace(".", "_")}.png')
            plt.close()

if __name__ == '__main__':
    checkpoint_path = 'checkpoints/checkpoint_epoch5.pth'
    analyze_checkpoint(checkpoint_path)
# -----------------------------
# 主函数
# -----------------------------
if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 检查输入路径
    if len(args.input) == 1 and os.path.isdir(args.input[0]):
        folder = args.input[0]
        in_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
        in_files.sort()
        out_dir = args.output or 'predictions'
        os.makedirs(out_dir, exist_ok=True)
        out_files = [os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + '_OUT.png') for f in in_files]
    else:
        in_files = args.input
        out_files = [args.output or (os.path.splitext(f)[0] + '_OUT.png') for f in in_files]

    # 模型加载
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    state_dict = torch.load(args.model, map_location=device)
    state_dict.pop('mask_values', None)

    # 添加权重检查
    print("Model state keys:", state_dict.keys())
    print("First conv layer weights stats:")
    first_conv = state_dict['inc.double_conv.0.weight']
    print(f"Mean: {first_conv.mean():.4f}, Std: {first_conv.std():.4f}")
    print(f"Min: {first_conv.min():.4f}, Max: {first_conv.max():.4f}")

    net.load_state_dict(state_dict)
    net.to(device=device)
    logging.info('Model loaded!')
    logging.info(f'Model parameters: {sum(p.numel() for p in net.parameters())}')
    logging.info(f'Model n_classes: {net.n_classes}')

    # VOC colormap
    cmap = voc_colormap(256)

    # 循环预测
    for i, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename).convert('RGB')
        
        # 添加输入图像信息
        logging.info(f'Input image size: {img.size}')
        
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        
        # 添加掩码信息
        logging.info(f'Mask shape: {mask.shape}')
        logging.info(f'Mask unique values: {np.unique(mask)}')
        
        if not args.no_save:
            out_filename = out_files[i]
            # 确保掩码值在正确范围内
            mask = mask.astype(np.uint8)
            color_mask = Image.fromarray(cmap[mask])
            color_mask.save(out_filename)
            logging.info(f'彩色Mask已保存至 {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)
