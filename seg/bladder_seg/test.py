import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 导入模型和数据集类
from model import UNetPlusPlus
from train import CustomSegmentationDataset

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# --------------------------
# 核心修复：Dice系数计算函数
# --------------------------
def compute_dice(preds, targets, num_classes, ignore_zero=False):
    """
    计算Dice系数，修复了空标签和维度处理问题

    参数:
        preds: 模型预测的类别索引 (batch_size, H, W)
        targets: 真实标签 (batch_size, H, W)
        num_classes: 类别数量
        ignore_zero: 是否忽略背景类(0)的计算
    """
    # 确保输入是numpy数组且形状一致
    preds = np.array(preds)
    targets = np.array(targets)
    assert preds.shape == targets.shape, f"预测与标签形状不匹配: {preds.shape} vs {targets.shape}"

    total_dice = 0.0
    valid_classes = 0  # 统计有真实标签的类别数

    for cls in range(num_classes):
        # 忽略背景类（如果需要）
        if ignore_zero and cls == 0:
            continue

        # 提取当前类别的掩码
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        # 计算交集和并集
        intersection = np.sum(pred_mask & target_mask)
        pred_sum = np.sum(pred_mask)
        target_sum = np.sum(target_mask)

        # 处理特殊情况：真实标签和预测均为空（全背景）
        if target_sum == 0:
            # 如果该类没有真实标签，跳过计算（不影响平均）
            continue

        valid_classes += 1

        # 处理预测为空但有真实标签的情况（Dice=0）
        if pred_sum == 0 and target_sum > 0:
            dice = 0.0
        # 处理分母为0的情况（理论上不会发生，因target_sum>0）
        elif (pred_sum + target_sum) == 0:
            dice = 1.0
        else:
            dice = 2.0 * intersection / (pred_sum + target_sum)

        total_dice += dice

    # 避免除以0（没有有效类别时返回0）
    return total_dice / valid_classes if valid_classes > 0 else 0.0


def compute_class_dice(preds, targets, num_classes):
    """计算每个类别的Dice系数，增加详细日志输出"""
    preds = np.array(preds)
    targets = np.array(targets)
    class_dice = []

    for cls in range(num_classes):
        pred_mask = (preds == cls)
        target_mask = (targets == cls)

        intersection = np.sum(pred_mask & target_mask)
        pred_sum = np.sum(pred_mask)
        target_sum = np.sum(target_mask)

        # 打印关键信息用于调试（目标类cls=1）
        if cls == 1:
            print(f"\n目标类(cls=1)统计:")
            print(f"  真实像素数: {target_sum}")
            print(f"  预测像素数: {pred_sum}")
            print(f"  交集中像素数: {intersection}")

        # 处理特殊情况
        if target_sum == 0:
            # 该类没有真实标签，Dice记为NaN（后续可过滤）
            class_dice.append(np.nan)
        elif (pred_sum + target_sum) == 0:
            class_dice.append(1.0)
        else:
            dice = 2.0 * intersection / (pred_sum + target_sum)
            class_dice.append(dice)

    return class_dice


def visualize_results(images, targets, preds, class_names, save_path, num_samples=5):
    """可视化测试结果，增加数据格式验证"""
    os.makedirs(save_path, exist_ok=True)

    # 反归一化参数
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 确保输入有效
    if len(images) == 0:
        print("警告：没有图像可用于可视化")
        return

    # 随机选择样本
    sample_count = min(num_samples, len(images))
    indices = np.random.choice(len(images), sample_count, replace=False)

    for i, idx in enumerate(indices):
        # 处理图像（反归一化）
        img = images[idx].cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)

        # 处理掩码（确保是numpy数组）
        target = np.array(targets[idx])
        pred = np.array(preds[idx])

        # 计算当前样本的Dice系数
        sample_dice = compute_dice(
            np.expand_dims(pred, axis=0),
            np.expand_dims(target, axis=0),
            len(class_names)
        )

        # 创建图像
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(img)
        axes[0].set_title('原始图像')
        axes[0].axis('off')

        axes[1].imshow(target, cmap='jet')
        axes[1].set_title(f'真实掩码 (类别值: {np.unique(target)})')
        axes[1].axis('off')

        axes[2].imshow(pred, cmap='jet')
        axes[2].set_title(f'预测掩码 (Dice: {sample_dice:.4f}, 类别值: {np.unique(pred)})')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'result_{i}.png'), dpi=300)
        plt.close()


def save_predicted_masks(preds, image_names, save_dir, img_size):
    """保存预测的掩码为图像文件"""
    os.makedirs(save_dir, exist_ok=True)

    for pred, name in zip(preds, image_names):
        # 将预测结果转换为0-255范围（便于可视化）
        max_val = pred.max()
        if max_val > 0:
            pred_scaled = (pred * (255 // max_val)).astype(np.uint8)
        else:
            pred_scaled = np.zeros_like(pred, dtype=np.uint8)
        # 保存为PNG文件
        mask_img = Image.fromarray(pred_scaled)
        mask_img.save(os.path.join(save_dir, f"{name}_pred_mask.png"))


def plot_dice_scores(class_names, class_dices, save_path):
    """绘制每个类别的Dice系数柱状图，处理NaN值"""
    plt.figure(figsize=(10, 6))
    x = np.arange(len(class_names))
    # 替换NaN值为0并标记
    class_dices_plot = [d if not np.isnan(d) else 0 for d in class_dices]
    bars = plt.bar(x, class_dices_plot, width=0.6)

    # 为NaN值的柱子添加特殊颜色
    for i, d in enumerate(class_dices):
        if np.isnan(d):
            bars[i].set_color('gray')

    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.ylabel('Dice系数')
    plt.title('每个类别的Dice系数')

    # 在柱状图上标注数值
    for i, v in enumerate(class_dices_plot):
        if np.isnan(class_dices[i]):
            plt.text(i, v + 0.01, '无数据', ha='center')
        else:
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'class_dice_scores.png'), dpi=300)
    plt.close()


def test(model, test_loader, device, num_classes, class_names, save_path):
    """测试模型并生成评估结果，增加数据验证步骤"""
    model.eval()

    all_preds = []
    all_targets = []
    all_images = []  # 存储图像张量（用于可视化）
    all_image_names = []
    current_idx = 0  # 用于跟踪当前批次的图像索引

    total_loss = 0.0
    total_dice = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        # 使用enumerate获取批次索引
        for batch_idx, (images, targets) in enumerate(tqdm(test_loader, unit="batch", desc="测试中")):
            # 计算当前批次的图像范围
            batch_size = images.size(0)
            end_idx = current_idx + batch_size

            # 获取当前批次的图像名称
            img_paths = test_loader.dataset.images[current_idx:end_idx]
            img_names = [os.path.splitext(os.path.basename(path))[0] for path in img_paths]
            all_image_names.extend(img_names)

            # 更新索引
            current_idx = end_idx

            images = images.to(device)
            targets = targets.to(device)

            # 模型预测
            outputs = model(images)
            loss = criterion(outputs, targets)

            # 计算预测结果（取概率最大的类别）
            preds = torch.argmax(outputs, dim=1)

            # 验证预测和标签的合理性（仅在第一批次打印）
            if batch_idx == 0:
                print(f"\n数据验证（第一批次）:")
                print(f"  模型输出形状: {outputs.shape}")
                print(f"  预测类别值范围: [{preds.min().item()}, {preds.max().item()}]")
                print(f"  真实标签值范围: [{targets.min().item()}, {targets.max().item()}]")

            # 计算Dice系数（转换为numpy数组进行计算）
            batch_preds = preds.cpu().numpy()
            batch_targets = targets.cpu().numpy()
            batch_dice = compute_dice(batch_preds, batch_targets, num_classes)

            # 累计指标
            total_loss += loss.item() * batch_size
            total_dice += batch_dice * batch_size

            # 保存结果用于后续分析
            all_preds.extend(batch_preds)
            all_targets.extend(batch_targets)
            all_images.extend(images)  # 保留张量用于可视化（需要反归一化）

    # 计算整体指标
    avg_loss = total_loss / len(test_loader.dataset)
    avg_dice = total_dice / len(test_loader.dataset)

    print(f"\n测试集整体指标:")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"平均Dice系数: {avg_dice:.4f}")

    # 计算每个类别的Dice系数
    class_dices = compute_class_dice(all_preds, all_targets, num_classes)

    # 打印每个类别的Dice
    print("\n每个类别的Dice系数:")
    for i, (cls_name, dice) in enumerate(zip(class_names, class_dices)):
        if np.isnan(dice):
            print(f"类别 {i} ({cls_name}): 无真实标签数据")
        else:
            print(f"类别 {i} ({cls_name}): {dice:.4f}")

    # 保存评估结果到文本文件
    with open(os.path.join(save_path, 'evaluation_results.txt'), 'w', encoding='utf-8') as f:
        f.write("测试集评估结果\n")
        f.write("====================\n")
        f.write(f"平均损失: {avg_loss:.4f}\n")
        f.write(f"平均Dice系数: {avg_dice:.4f}\n\n")
        f.write("每个类别的Dice系数:\n")
        for i, (cls_name, dice) in enumerate(zip(class_names, class_dices)):
            if np.isnan(dice):
                f.write(f"类别 {i} ({cls_name}): 无真实标签数据\n")
            else:
                f.write(f"类别 {i} ({cls_name}): {dice:.4f}\n")

    # 可视化结果
    visualize_results(
        all_images,
        all_targets,
        all_preds,
        class_names,
        os.path.join(save_path, '可视化结果')
    )

    # 绘制类别Dice柱状图
    plot_dice_scores(class_names, class_dices, save_path)

    # 获取图像尺寸
    img_size = test_loader.dataset.transform.transforms[0].height

    # 保存预测的掩码
    save_predicted_masks(
        all_preds,
        all_image_names,
        os.path.join(save_path, '预测掩码'),
        img_size
    )

    return avg_loss, avg_dice, class_dices


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试YOLOv12+UNet++分割模型')
    parser.add_argument('--data-root', type=str, default='./dataset',
                        help='数据集根目录')
    parser.add_argument('--model-path', type=str, default="models/best_model.pth",
                        help='训练好的模型路径（必填）')
    parser.add_argument('--save-path', type=str, default='./test_results',
                        help='测试结果保存目录')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='类别数量（默认2类）')
    parser.add_argument('--class-names', type=str, nargs='+', default=['背景', '目标'],
                        help='类别名称列表（默认：["背景", "目标"]）')
    parser.add_argument('--img-size', type=int, default=512,
                        help='输入图像尺寸（默认512）')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='批次大小（默认4）')
    parser.add_argument('--workers', type=int, default=0,  # 修复Windows多线程问题
                        help='数据加载线程数（Windows建议设为0）')
    parser.add_argument('--device', type=str, default=None,
                        help='使用的设备（cuda或cpu）')
    args = parser.parse_args()

    # 创建保存结果的目录
    os.makedirs(args.save_path, exist_ok=True)

    # 设备设置
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 验证类别名称数量是否正确
    if len(args.class_names) != args.num_classes:
        raise ValueError(f"类别名称数量 ({len(args.class_names)}) 必须与类别数量 ({args.num_classes}) 一致")

    # 数据转换（与训练时保持一致）
    test_transform = A.Compose([
        A.Resize(height=args.img_size, width=args.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # 加载测试数据集
    test_dataset = CustomSegmentationDataset(
        dataset_root=args.data_root,
        split='test',
        transform=test_transform,
        num_classes=args.num_classes
    )

    # 验证测试集是否包含目标类
    print("\n测试集数据验证:")
    sample_mask = np.array(Image.open(
        os.path.join(test_dataset.mask_dir,
                     f"{os.path.splitext(os.path.basename(test_dataset.images[0]))[0]}_mask.png")
    ).convert('L'))
    sample_mask[sample_mask == 255] = 1  # 应用与训练集相同的转换
    print(f"  测试集样本掩码的类别值: {np.unique(sample_mask)}")
    if 1 not in np.unique(sample_mask) and args.num_classes > 1:
        print("  警告：测试集样本中未发现目标类(1)的标签！")

    # 创建数据加载器（Windows系统建议num_workers=0）
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # 初始化模型并加载权重
    model = UNetPlusPlus(num_classes=args.num_classes).to(device)

    # 加载模型时设置weights_only=False以兼容旧版本模型
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\n已加载模型: {args.model_path}")
    print(f"模型训练时的最佳Dice系数: {checkpoint.get('best_dice', 0):.4f}")

    # 开始测试
    print(f"\n开始测试，测试集样本数: {len(test_dataset)}")
    test_loss, test_dice, class_dices = test(
        model, test_loader, device, args.num_classes, args.class_names, args.save_path
    )

    print(f"\n测试完成，平均Dice系数: {test_dice:.4f}")
    print(f"测试结果已保存到: {args.save_path}")


if __name__ == "__main__":
    main()
