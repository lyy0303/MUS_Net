import os
import torch
import logging
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from models.model import LLNM_Net
from dataloader import get_dataloader


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(train_loader, desc='Training'):
        # 准备输入数据
        image = batch['image'].to(device)
        tumor_marker = batch['tumor_marker'].to(device)
        other_clinical = batch['other_clinical'].to(device)
        img_fea = batch['img_fea'].to(device)
        labels = batch['label'].to(device)

        # 前向传播
        optimizer.zero_grad()
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
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(preds.flatten())
        all_labels.extend(labels.detach().cpu().numpy())

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    # 计算指标
    avg_loss = total_loss / len(train_loader)
    preds_binary = np.round(all_preds)
    acc = accuracy_score(all_labels, preds_binary)
    precision = precision_score(all_labels, preds_binary, zero_division=0)
    recall = recall_score(all_labels, preds_binary, zero_division=0)
    f1 = f1_score(all_labels, preds_binary, zero_division=0)
    auc = roc_auc_score(all_labels, all_preds)

    return avg_loss, acc, precision, recall, f1, auc


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # 准备输入数据
            image = batch['image'].to(device)
            tumor_marker = batch['tumor_marker'].to(device)
            age = batch['other_clinical'].to(device)
            img_fea = batch['img_fea'].to(device)
            labels = batch['label'].to(device)

            # 前向传播
            outputs = model(
                x=image,
                img_fea=img_fea,
                tumor_marker=tumor_marker,
                other_clinical=age,
                labels=labels
            )

            # 计算损失
            loss = outputs['loss']
            total_loss += loss.item()

            # 记录预测结果
            logits = outputs['logits']
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    avg_loss = total_loss / len(val_loader)
    preds_binary = np.round(all_preds)
    acc = accuracy_score(all_labels, preds_binary)
    precision = precision_score(all_labels, preds_binary, zero_division=0)
    recall = recall_score(all_labels, preds_binary, zero_division=0)
    f1 = f1_score(all_labels, preds_binary, zero_division=0)
    auc = roc_auc_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, preds_binary)

    return avg_loss, acc, precision, recall, f1, auc, cm


def main(args):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )

    # 创建数据加载器
    try:
        train_loader = get_dataloader(
            data_dir=args.data_dir,
            excel_path=args.excel_path,
            split='train',
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )

        val_loader = get_dataloader(
            data_dir=args.data_dir,
            excel_path=args.excel_path,
            split='val',
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    except ValueError as e:
        logging.error(f"数据加载错误: {str(e)}")
        return

    # 创建模型
    model = LLNM_Net().to(device)

    # 优化器和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = torch.nn.BCEWithLogitsLoss()  # 二分类损失函数

    # 训练参数
    best_val_auc = 0.0

    # 开始训练
    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss, train_acc, train_precision, train_recall, train_f1, train_auc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        logging.info(
            f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}, "
            f"精确率: {train_precision:.4f}, 召回率: {train_recall:.4f}, "
            f"F1: {train_f1:.4f}, AUC: {train_auc:.4f}"
        )

        # 验证
        val_loss, val_acc, val_precision, val_recall, val_f1, val_auc, val_cm = validate(
            model, val_loader, criterion, device
        )
        logging.info(
            f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}, "
            f"精确率: {val_precision:.4f}, 召回率: {val_recall:.4f}, "
            f"F1: {val_f1:.4f}, AUC: {val_auc:.4f}"
        )
        logging.info(f"验证集混淆矩阵:\n{val_cm}")

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc
            }, os.path.join(args.save_dir, 'best_model.pth'))
            logging.info(f"保存最佳模型 (AUC: {best_val_auc:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练医学图像二分类模型')

    # 数据参数
    parser.add_argument('--data_dir', default="dataset", help='数据集根目录')
    parser.add_argument('--excel_path', default="dataset/clinical_data.xlsx", help='临床数据Excel文件路径')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载线程数')

    # 保存参数
    parser.add_argument('--save_dir', default='checkpoints', help='模型保存目录')
    parser.add_argument('--log_dir', default='logs', help='日志保存目录')

    args = parser.parse_args()
    main(args)
