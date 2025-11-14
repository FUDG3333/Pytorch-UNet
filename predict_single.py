import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import numpy as np
import logging
import matplotlib.pyplot as plt
import cv2

from unet import UNet
from utils.data_loading import BasicDataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def get_voc_colormap():
    """获取VOC数据集的调色板"""
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    
    cmap = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    return cmap

def predict_img(model, full_img, device, scale_factor=0.5):
    model.eval()
    img = BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False)
    img = torch.from_numpy(img).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        logits = model(img)  # [1, C, h', w']
        logits_up = F.interpolate(logits, size=(full_img.size[1], full_img.size[0]), 
                                  mode='bilinear', align_corners=False)
        probs = F.softmax(logits_up, dim=1)[0].cpu().numpy()  # C,H,W
        mask = probs.argmax(axis=0).astype(np.uint8)  # H,W
    
    return mask, probs

def visualize_results(original_img, mask, probs, n_classes=21):
    """可视化原图、预测掩码、叠加结果和类别占比"""
    cmap = get_voc_colormap()
    colored_mask = cmap[mask]
    
    # 创建叠加图像
    overlay = np.array(original_img).copy()
    alpha = 0.5
    overlay = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
    
    # 计算每类像素占比
    unique, counts = np.unique(mask, return_counts=True)
    total = mask.size
    class_percentages = np.zeros(n_classes)
    for u, c in zip(unique, counts):
        class_percentages[u] = c / total * 100
    
    # 创建 4 个子图：原图、掩码、叠加、直方图
    fig = plt.figure(figsize=(16, 12))
    
    # 原图
    ax1 = plt.subplot(2, 2, 1)
    ax1.imshow(original_img)
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # 预测掩码（彩色）
    ax2 = plt.subplot(2, 2, 2)
    ax2.imshow(colored_mask)
    ax2.set_title('Predicted Segmentation Mask', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # 叠加图
    ax3 = plt.subplot(2, 2, 3)
    ax3.imshow(overlay)
    ax3.set_title('Overlay (50% transparency)', fontsize=14, fontweight='bold')
    ax3.axis('off')
    
    # 类别占比直方图
    ax4 = plt.subplot(2, 2, 4)
    colors_bar = cmap[:n_classes] / 255.0  # 归一化到 0-1
    bars = ax4.bar(range(n_classes), class_percentages, color=colors_bar)
    ax4.set_xlabel('Class', fontsize=12)
    ax4.set_ylabel('Percentage (%)', fontsize=12)
    ax4.set_title('Per-class Pixel Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlim(-0.5, n_classes - 0.5)
    
    # 设置横轴只显示 0 到 20 的整数
    ax4.set_xticks(range(n_classes))
    ax4.set_xticklabels([str(i) for i in range(n_classes)], fontsize=10)
    
    # 仅在非零类上标注百分比
    for i, (u, c) in enumerate(zip(unique, counts)):
        if class_percentages[u] > 0.5:  # 仅显示占比 > 0.5% 的类
            ax4.text(u, class_percentages[u] + 1, f'{class_percentages[u]:.1f}%', 
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息到日志
    logging.info('\n' + '='*60)
    logging.info('Prediction Summary:')
    logging.info('='*60)
    logging.info('Predicted classes with pixel count and percentage:')
    for u, c in zip(unique, counts):
        logging.info(f'  Class {u:2d}: {c:8d} pixels ({class_percentages[u]:6.2f}%)')
    logging.info('='*60)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path('./checkpoints/checkpoint_epoch15.pth')
    img_path = Path('data/VOCdevkit/VOC2012/JPEGImages/2007_000175.jpg')
    output_dir = Path('./test_results')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化模型
    model = UNet(n_channels=3, n_classes=21, bilinear=False)
    model.to(device)
    
    # 加载 checkpoint
    logging.info(f'Loading checkpoint from {model_path}')
    ckpt = torch.load(str(model_path), map_location=device)
    
    # 提取 model_state
    if 'model_state' in ckpt:
        state_dict = ckpt['model_state']
        logging.info('Found "model_state" in checkpoint')
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
        logging.info('Found "state_dict" in checkpoint')
    else:
        state_dict = ckpt
        logging.info('Using checkpoint as direct state_dict')
    
    # 加载权重
    try:
        model.load_state_dict(state_dict, strict=True)
        logging.info('✅ Model weights loaded successfully (strict=True)')
    except RuntimeError as e:
        logging.warning(f'Strict load failed, retrying with strict=False...')
        res = model.load_state_dict(state_dict, strict=False)
        logging.info(f'Load result: {res}')
    
    # 预测
    img = Image.open(img_path).convert('RGB')
    logging.info(f'Processing image: {img_path.name} (size: {img.size})')
    mask, probs = predict_img(model, img, device, scale_factor=0.5)
    
    # 保存掩码到文件
    out_path = output_dir / f'{img_path.stem}_pred_mask.png'
    Image.fromarray(mask).save(str(out_path))
    logging.info(f'✅ Predicted mask saved to {out_path}')
    
    # 显示可视化窗口
    logging.info('Displaying visualization window...')
    visualize_results(img, mask, probs, n_classes=21)
