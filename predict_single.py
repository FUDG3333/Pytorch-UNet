import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from unet import UNet
from utils.data_loading import BasicDataset


def predict_img(model, full_img, device, scale_factor=0.5, out_threshold=0.5):
    model.eval()
    # 注意这里加了 None 作为 mask_values 占位参数
    img = BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    img = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = model(img)
        if model.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
            full_mask = probs.argmax(dim=0)
        else:
            probs = torch.sigmoid(output)[0]
            full_mask = (probs > out_threshold).float()
    return full_mask.cpu()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './checkpoints/best_model.pth'
    # model_path = './checkpoints/checkpoint_epoch5.pth'
    img_path = Path('data/VOCdevkit/VOC2012/JPEGImages/2007_000170.jpg')
    output_dir = Path('./test_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化模型
    model = UNet(n_channels=3, n_classes=21, bilinear=False)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    # 读取图像并预测
    img = Image.open(img_path).convert('RGB')
    mask = predict_img(model=model, full_img=img, device=device, scale_factor=0.5)

    # 保存灰度掩码
    mask_np = mask.numpy().astype(np.uint8)
    mask_img = Image.fromarray(mask_np)
    out_path = output_dir / f'{img_path.stem}_mask.png'
    mask_img.save(out_path)

    print(f'✅ Saved mask to {out_path}')

    # 可视化原图和预测结果
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.title('Original')
    plt.subplot(1,2,2)
    plt.imshow(mask_np, cmap='gray')
    plt.title('Predicted Mask')
    plt.show()
