import os
import torch
import logging
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
from models.model import LLNM_Net
from dataloader import get_dataloader

# 设置中文字体，确保图表中文正常显示
plt.rcParams["font.family"] = ["Times New Roman"]
# plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.unicode_minus"] = False


def test(model, test_loader, criterion, device):
    """在测试集上评估模型性能"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_preds_proba = []  # 存储概率值用于AUC计算
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # 准备输入数据
            image = batch['image'].to(device)
            tumor_marker = batch['tumor_marker'].to(device)
            other_clinical = batch['other_clinical'].to(device)
            img_fea = batch['img_fea'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            outputs = model(
                x=image,
                img_fea=img_fea,
                tumor_marker=tumor_marker,
                other_clinical=other_clinical,
                labels=labels
            )

            # 计算损失
            loss = outputs['loss']
            total_loss += loss.item()

            # 记录预测结果
            logits = outputs['logits']
            preds_proba = torch.sigmoid(logits).cpu().numpy()  # 概率值
            preds = np.round(preds_proba)  # 二值化预测结果

            all_preds.extend(preds.flatten())
            all_preds_proba.extend(preds_proba.flatten())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    avg_loss = total_loss / len(test_loader)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_preds_proba)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, zero_division=0)
    print(f'all_preds:{all_labels[2], all_labels[7], all_labels[35]}')
    print(f'all_preds:{all_preds[2], all_preds[7], all_preds[35]}')
    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'probabilities': all_preds_proba,
        'labels': all_labels
    }

def plot_confusion_matrix(cm, save_path):
    """绘制混淆矩阵热力图"""
    """绘制更美观、大字体的混淆矩阵热力图"""
    # 设置图片清晰度（可选，也可在savefig时设置dpi）
    plt.rcParams['figure.dpi'] = 300
    # 增大全局字体大小
    plt.rcParams['font.size'] = 18
    plt.figure(figsize=(8, 6))  # 可根据需要调整尺寸

    # 绘制热力图，调整annot字体、添加边框、优化颜色条
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='PuBu',  # Pastel1_r,PuBu
                xticklabels=['Positive', 'Negative'],
                yticklabels=['Cancer', 'No cancer'],
                annot_kws={"size": 22},  # 调整标注数字的字体大小
                linewidths=0.5,  # 添加格子边框
                cbar_kws={"shrink": 0.8}  # 调整颜色条大小
                )

    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('Actual label', fontsize=20)
    plt.title('Confusion matrix', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')  # bbox_inches避免标签被截断
    plt.close()

def plot_roc_curve(labels, probabilities, auc, save_path):
    labels = np.array(labels)
    probabilities = np.array(probabilities)

    fpr, tpr, _ = roc_curve(labels, probabilities)

    # 统一字体设置（与混淆矩阵保持一致）
    plt.rcParams.update({
        'font.size': 18,
        'axes.linewidth': 1,
        'axes.edgecolor': 'gray'
    })

    plt.figure(figsize=(8, 6))  # 稍大画布避免拥挤

    # 绘制ROC曲线（加粗线条，优化颜色）
    plt.plot(
        fpr, tpr,
        color='#2c7fb8',  # 更专业的蓝色（非默认橙色）
        lw=3,  # 线条加粗
        label=f'ROC curve (AUC = {auc:.4f})'
    )

    # 绘制对角线（参考线）
    plt.plot(
        [0, 1], [0, 1],
        color='#999999',  # 浅灰色更柔和
        lw=2,
        linestyle='--',
        label='Random Classifier'
    )

    # 坐标轴范围与刻度优化
    plt.xlim([-0.01, 1.01])  # 稍留边距更美观
    plt.ylim([-0.01, 1.06])
    plt.xticks(np.arange(0, 1.1, 0.2), fontsize=20)  # 间隔0.2更清晰
    plt.yticks(np.arange(0, 1.1, 0.2), fontsize=20)

    # 坐标轴标签与标题（加粗+大字体）
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('ROC Curve for Cancer Detection', fontsize=20)

    # 图例优化
    plt.legend(
        loc="lower right",
        fontsize=18,
        frameon=True,  # 显示图例边框
        edgecolor='gray'  # 图例边框颜色
    )

    # 网格线增强可读性
    plt.grid(alpha=0.3, linestyle='--')  # 浅灰色虚线网格

    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(labels, probabilities, save_path):
    labels = np.array(labels)
    probabilities = np.array(probabilities)

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.close()


def plot_probability_distribution(labels, probabilities, save_path):
    labels_np = np.array(labels)
    probabilities_np = np.array(probabilities)

    plt.figure(figsize=(8, 6))
    # 正样本概率分布（筛选真实标签为1的样本）
    sns.histplot(probabilities_np[labels_np == 1],
                 bins=20, kde=True, label='Positive', color='red', alpha=0.6)
    # 负样本概率分布（筛选真实标签为0的样本）
    sns.histplot(probabilities_np[labels_np == 0],
                 bins=20, kde=True, label='Negative', color='blue', alpha=0.6)
    plt.axvline(x=0.5, color='green', linestyle='--', label='Decision threshold (0.5)')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Sample Count')
    plt.title('Predicted Probability Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=500)
    plt.close()


def plot_metrics_bar(metrics, save_path):
    """绘制各评估指标的柱状图"""
    # 统一字体和样式设置
    plt.rcParams.update({
        'font.size': 12,
        'axes.edgecolor': '#808080'  # #4C7780
    })

    plt.figure(figsize=(10, 6))  # 稍大画布提升美观度

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC']
    metrics_values = [
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1'],
        metrics['auc']
    ]

    # 绘制柱状图（使用与ROC曲线协调的配色）
    sns.barplot(
        x=metrics_names,
        y=metrics_values,
        palette=['#2171B5', '#08519C', '#C6DBEF', '#2171B5', '#08519C'],  # 渐变色系
        edgecolor='white',  # 柱形边框
        linewidth=1,  # 边框线宽
        width=0.7
    )

    # 坐标轴范围与刻度
    plt.ylim(0, 1.1)  # 留足顶部空间放置数值
    plt.yticks(np.arange(0.0, 1.1, 0.2), fontsize=18)  # 与ROC曲线保持一致的刻度间隔
    plt.xticks(fontsize=20, rotation=0)  # x轴标签不旋转

    # 在柱状图上添加数值（加大字体并加粗）
    for i, v in enumerate(metrics_values):
        plt.text(
            i, v + 0.03,  # 位置微调，避免紧贴柱子
            f'{v:.4f}',
            ha='center',
            fontsize=20,
            # weight='bold',  # 数值加粗更醒目
            color='#333333'  # 深灰色数值更易读
        )

    # 坐标轴标签与标题（统一字体大小和加粗）
    plt.xlabel('Evaluation Metrics', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.title('Model Evaluation Metrics', fontsize=20)

    # 添加网格线增强可读性
    plt.grid(axis='y', alpha=0.3, linestyle='--')  # 仅y轴网格线

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(args):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建保存目录（确保图表和结果目录存在）
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    plots_dir = os.path.join(args.save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'test.log')),
            logging.StreamHandler()
        ]
    )

    # 创建测试数据加载器
    try:
        test_loader = get_dataloader(
            data_dir=args.data_dir,
            excel_path=args.excel_path,
            split='test',  # 使用测试集
            batch_size=args.batch_size,
            shuffle=False,  # 测试时不打乱顺序
            num_workers=args.num_workers
        )
    except ValueError as e:
        logging.error(f"数据加载错误: {str(e)}")
        return

    # 创建模型并加载权重
    model = LLNM_Net().to(device)

    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        logging.error(f"模型文件不存在: {args.model_path}")
        return

    # 加载模型权重
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"成功加载模型: {args.model_path}")
        logging.info(f"模型训练轮数: {checkpoint.get('epoch', '未知')}")
        logging.info(f"模型验证AUC: {checkpoint.get('val_auc', '未知'):.4f}")
    except Exception as e:
        logging.error(f"模型加载失败: {str(e)}")
        return

    # 定义损失函数
    criterion = torch.nn.BCEWithLogitsLoss()

    # 在测试集上评估
    logging.info("开始测试...")
    results = test(model, test_loader, criterion, device)

    # 输出测试结果
    logging.info("\n===== 测试集性能指标 =====")
    logging.info(f"损失: {results['loss']:.4f}")
    logging.info(f"准确率: {results['accuracy']:.4f}")
    logging.info(f"精确率: {results['precision']:.4f}")
    logging.info(f"召回率: {results['recall']:.4f}")
    logging.info(f"F1分数: {results['f1']:.4f}")
    logging.info(f"AUC: {results['auc']:.4f}")

    logging.info("\n混淆矩阵:")
    logging.info(f"{results['confusion_matrix']}")

    logging.info("\n分类报告:")
    logging.info(f"{results['classification_report']}")

    # 生成并保存图表（修复后可正常运行）
    logging.info("\n生成评估图表...")
    plot_confusion_matrix(
        results['confusion_matrix'],
        os.path.join(plots_dir, 'confusion_matrix.png')
    )

    plot_roc_curve(
        results['labels'],
        results['probabilities'],
        results['auc'],
        os.path.join(plots_dir, 'roc_curve.png')
    )

    plot_precision_recall_curve(
        results['labels'],
        results['probabilities'],
        os.path.join(plots_dir, 'precision_recall_curve.png')
    )

    plot_probability_distribution(
        results['labels'],
        results['probabilities'],
        os.path.join(plots_dir, 'probability_distribution.png')
    )

    plot_metrics_bar(
        results,
        os.path.join(plots_dir, 'metrics_bar.png')
    )

    logging.info(f"评估图表已保存至: {plots_dir}")

    # 保存预测结果（可选）
    if args.save_results:
        results_dir = os.path.join(args.save_dir, 'test_results')
        os.makedirs(results_dir, exist_ok=True)

        np.savez(os.path.join(results_dir, 'predictions.npz'),
                 predictions=results['predictions'],
                 probabilities=results['probabilities'],
                 labels=results['labels'])
        logging.info(f"预测结果已保存至: {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='医学图像二分类模型测试脚本')

    # 数据参数
    parser.add_argument('--data_dir', default="processed_dataset", help='数据集根目录')
    parser.add_argument('--excel_path', default="processed_dataset/clinical_data.xlsx", help='临床数据Excel文件路径')

    # 测试参数
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')

    # 模型参数
    parser.add_argument('--model_path', default='checkpoints/best_model.pth',
                        help='训练好的模型路径')

    # 保存参数
    parser.add_argument('--save_dir', default='results0', help='测试结果保存目录')
    parser.add_argument('--log_dir', default='logs', help='日志保存目录')
    parser.add_argument('--save_results', action='store_true', help='是否保存预测结果')

    args = parser.parse_args()
    main(args)