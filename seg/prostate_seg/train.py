import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
# 修改模型导入
from model import UNetPlusPlus  # 从YOLOv12UNetPlusPlus改为UNetPlusPlus

plt.rcParams["font.family"] = ["Times New Roman"]
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 自定义数据集类（保持不变）
class CustomSegmentationDataset(Dataset):
    def __init__(self, dataset_root, split, transform=None, num_classes=2):
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform
        self.num_classes = num_classes

        # 定义图像和掩码路径
        self.img_dir = os.path.join(dataset_root, split, 'images')
        self.mask_dir = os.path.join(dataset_root, split, 'masks')

        # 验证路径存在
        assert os.path.exists(self.img_dir), f"图像目录不存在: {self.img_dir}"
        assert os.path.exists(self.mask_dir), f"掩码目录不存在: {self.mask_dir}"

        # 获取所有图像文件
        self.images = [
            os.path.join(self.img_dir, f)
            for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        # 过滤没有对应掩码的图像
        self._filter_valid_images()

        assert len(self.images) > 0, f"{split}集中未找到有效图像"

    def _filter_valid_images(self):
        valid_images = []
        for img_path in self.images:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(self.mask_dir, f"{img_name}_mask.png")
            if os.path.exists(mask_path):
                valid_images.append(img_path)
            else:
                print(f"警告: 未找到{img_path}对应的掩码，已跳过")
        self.images = valid_images

    def __getitem__(self, idx):
        # 读取图像
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        # 读取掩码
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(self.mask_dir, f"{img_name}_mask.png")
        mask = Image.open(mask_path).convert('L')  # 转为单通道

        # 转换掩码: 将255转为1（针对二值分割）
        mask_np = np.array(mask, dtype=np.int64)
        mask_np[mask_np == 255] = 1

        # 确保标签在有效范围内
        mask_np = np.clip(mask_np, 0, self.num_classes - 1)

        # 应用数据转换
        if self.transform:
            augmented = self.transform(image=np.array(image), mask=mask_np)
            image = augmented['image']
            mask = augmented['mask'].long()  # 确保为长整数类型

        return image, mask

    def __len__(self):
        return len(self.images)


# 计算Dice系数（保持不变）
def compute_dice(preds, targets, num_classes):
    """
    preds: 模型输出的概率图 (batch_size, num_classes, H, W)
    targets: 真实标签 (batch_size, H, W)
    """
    preds = np.argmax(preds, axis=1)  # 转换为类别索引
    total_dice = 0.0

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask) + np.sum(target_mask)

        if union == 0:
            dice = 0.0
        else:
            dice = 2.0 * intersection / union

        total_dice += dice

    return total_dice / num_classes


# 计算每个类别的Dice系数（保持不变）
def compute_class_dice(preds, targets, num_classes):
    preds = np.argmax(preds, axis=1)
    class_dice = []

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask) + np.sum(target_mask)

        if union == 0:
            dice = 0.0
        else:
            dice = 2.0 * intersection / union

        class_dice.append(dice)

    return class_dice


# 训练一个epoch（保持不变）
def train_one_epoch(model, train_loader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    with tqdm(train_loader, unit="batch") as t:
        t.set_description("Training")

        for images, targets in t:
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算指标
            batch_dice = compute_dice(outputs.cpu().detach().numpy(),
                                      targets.cpu().numpy(), num_classes)

            total_loss += loss.item() * images.size(0)
            total_dice += batch_dice * images.size(0)

            t.set_postfix(loss=loss.item(), dice=batch_dice)

    avg_loss = total_loss / len(train_loader.dataset)
    avg_dice = total_dice / len(train_loader.dataset)

    return avg_loss, avg_dice


# 验证（保持不变）
def validate(model, val_loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as t:
            t.set_description("Validation")

            for images, targets in t:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                # 计算指标
                batch_dice = compute_dice(outputs.cpu().numpy(),
                                          targets.cpu().numpy(), num_classes)

                total_loss += loss.item() * images.size(0)
                total_dice += batch_dice * images.size(0)

                t.set_postfix(loss=loss.item(), dice=batch_dice)

    avg_loss = total_loss / len(val_loader.dataset)
    avg_dice = total_dice / len(val_loader.dataset)

    return avg_loss, avg_dice


# 主函数（主要修改模型初始化部分）
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练UNet++分割模型')
    parser.add_argument('--data-root', type=str, default='./dataset',
                        help='数据集根目录')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='类别数量（默认2类）')
    parser.add_argument('--img-size', type=int, default=512,
                        help='输入图像尺寸（默认512）')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批次大小（默认4）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数（默认50）')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='学习率（默认1e-4）')
    parser.add_argument('--weight-decay', type=float, default=5e-5,
                        help='权重衰减（默认1e-5）')
    parser.add_argument('--workers', type=int, default=4,
                        help='数据加载线程数（默认4）')
    parser.add_argument('--save-dir', type=str, default='./models',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--use-pretrained', action='store_true', default=True,
                        help='是否使用预训练权重')
    parser.add_argument('--no-pretrained', dest='use_pretrained', action='store_false',
                        help='不使用预训练权重')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据转换
    train_transform = A.Compose([
        A.Resize(height=args.img_size, width=args.img_size),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(height=args.img_size, width=args.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 加载数据集
    train_dataset = CustomSegmentationDataset(
        dataset_root=args.data_root,
        split='train',
        transform=train_transform,
        num_classes=args.num_classes
    )

    val_dataset = CustomSegmentationDataset(
        dataset_root=args.data_root,
        split='val',
        transform=val_transform,
        num_classes=args.num_classes
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # 初始化模型 - 主要修改点
    model = UNetPlusPlus(
        num_classes=args.num_classes,
        use_pretrained=args.use_pretrained
    ).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10
    )

    # 恢复训练
    start_epoch = 0
    best_dice = 0.0

    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint['best_dice']
        print(f"从 {args.resume} 恢复训练，起始轮次: {start_epoch}")

    # 训练历史记录
    history = {
        'train_loss': [], 'train_dice': [],
        'val_loss': [], 'val_dice': []
    }

    # 开始训练
    print(f"开始训练，共 {args.epochs} 轮，训练集: {len(train_dataset)} 样本，验证集: {len(val_dataset)} 样本")
    print(f"使用预训练权重: {args.use_pretrained}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.num_classes
        )

        # 验证
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, args.num_classes
        )

        # 更新学习率
        scheduler.step(val_dice)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        # 打印 epoch 结果
        print(f"训练损失: {train_loss:.4f}, 训练Dice: {train_dice:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证Dice: {val_dice:.4f}")
        print(f"当前学习率: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            checkpoint_path = os.path.join(args.save_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'args': vars(args)
            }, checkpoint_path)
            print(f"已保存最佳模型到 {checkpoint_path}，Dice: {best_dice:.4f}")

        # 每10轮保存一次 checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'args': vars(args)
            }, checkpoint_path)

    # 训练结束后绘制指标曲线
    plt.figure(figsize=(12, 4))  # 适当加大高度，避免字体拥挤

    # 统一字体设置（全局字体大小18）
    plt.rcParams.update({
        'font.size': 16,
        'axes.edgecolor': 'gray'
    })

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    # 训练曲线：深蓝色+加粗；验证曲线：橙色+加粗（对比鲜明）
    plt.plot(history['train_loss'], label='Training', color='#2c7fb8', linewidth=2)
    plt.plot(history['val_loss'], label='Validation', color='#e41a1c', linewidth=2)
    plt.title('Loss curve')  # 标题加粗
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(frameon=True, edgecolor='gray', fontsize=14)  # 图例稍小避免拥挤
    plt.grid(alpha=0.3, linestyle='--')  # 添加网格线

    # 绘制Dice系数曲线
    plt.subplot(1, 2, 2)
    # 保持配色一致性：训练曲线深蓝色，验证曲线橙色
    plt.plot(history['train_dice'], label='Training', color='#2c7fb8', linewidth=2)
    plt.plot(history['val_dice'], label='Validation', color='#e41a1c', linewidth=2)
    plt.title('Dice coefficient curve')  # 标题加粗
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend(frameon=True, edgecolor='gray', fontsize=14)
    plt.grid(alpha=0.3, linestyle='--')

    plt.tight_layout()
    # 保存时确保标签完整
    plt.savefig(os.path.join(args.save_dir, 'training_metrics_0_dpi500(0.00005,0.5,10).png'),
                dpi=500, bbox_inches='tight')
    plt.close()

    print(f"训练完成，最佳验证Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()