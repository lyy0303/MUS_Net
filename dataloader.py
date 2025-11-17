import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 图像预处理（保持不变）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ProstateDataset(Dataset):
    def __init__(self, data_dir, excel_path, split="train"):
        self.base_dir = os.path.join(data_dir, split)
        self.split = split
        self.transform = transform

        # 1. 读取Excel并预处理ID
        self.clinical_df = pd.read_excel(excel_path)
        self.clinical_df['ID'] = self.clinical_df['ID'].astype(str).str.strip()

        # 2. 图像文件夹配置（prostate=JPG，其他=PNG）
        self.image_folders = [("prostate", ".jpg"), ("mosaic_image", ".png"), ("distance_map", ".png")]

        # 3. 获取有效图像ID（确保三类图像都存在）
        self.valid_ids = self._get_valid_image_ids()

        # 4. 合并临床数据（只保留有图像的样本）
        self.merged_df = self._merge_clinical_data()

        # 5. 关键修复1：处理数值特征+强制标签为0/1（消除非法标签）
        self._clean_numeric_features()

        # 6. 最终检查（确保无空样本）
        if len(self.merged_df) == 0:
            raise ValueError(f"{split}集无有效样本，请检查数据路径或格式")

    def _get_valid_image_ids(self):
        # 从prostate文件夹（JPG）提取基础ID
        first_folder, first_ext = self.image_folders[0]
        first_folder_path = os.path.join(self.base_dir, first_folder)

        if not os.path.exists(first_folder_path):
            raise NotADirectoryError(f"图像文件夹不存在：{first_folder_path}")

        # 提取ID（去掉.jpg后缀）
        all_ids = [os.path.splitext(f)[0] for f in os.listdir(first_folder_path) if f.endswith(first_ext)]

        # 验证其他文件夹（PNG）是否有对应图像
        valid_ids = []
        for sample_id in all_ids:
            if all(os.path.exists(os.path.join(self.base_dir, folder, f"{sample_id}{ext}"))
                   for folder, ext in self.image_folders[1:]):
                valid_ids.append(sample_id)

        print(f"{self.split}集找到{len(valid_ids)}个完整样本（三类图像均存在）")
        return valid_ids

    def _merge_clinical_data(self):
        # 只保留有图像的样本（避免无效数据）
        merged_df = self.clinical_df[self.clinical_df['ID'].isin(self.valid_ids)].copy()

        # 提示Excel中缺失的ID（可选）
        missing_in_excel = [id for id in self.valid_ids if id not in merged_df['ID'].values]
        if missing_in_excel:
            print(f"警告：{self.split}集有{len(missing_in_excel)}个图像ID无临床数据（示例：{missing_in_excel[:3]}）")

        return merged_df

    def _clean_numeric_features(self):
        """处理数值特征+强制标签为0/1（核心修复）"""
        # 需要处理的数值列（临床指标+标签）
        numeric_cols = ['CA199', 'tPSA', 'fPSA', 'cPSA', 'age', 'label']

        # 逐个处理列（彻底消除pandas inplace警告）
        for col in numeric_cols:
            # 步骤1：转换为数值类型（无法转换的设为NaN）
            self.merged_df[col] = pd.to_numeric(self.merged_df[col], errors='coerce')

            # 步骤2：处理缺失值（用列平均值填充，非inplace写法）
            na_count = self.merged_df[col].isna().sum()
            if na_count > 0:
                mean_val = self.merged_df[col].mean()
                self.merged_df[col] = self.merged_df[col].fillna(mean_val)  # 替换inplace=True
                print(f"警告：{self.split}集{col}列有{na_count}个缺失值，已用平均值（{mean_val:.2f}）填充")

        # 步骤3：强制标签为0/1（解决CUDA断言错误的核心！）
        # 情况1：标签是连续值（如概率）→ 用0.5阈值二值化
        if self.merged_df['label'].min() >= 0 and self.merged_df['label'].max() <= 1:
            self.merged_df['label'] = (self.merged_df['label'] >= 0.5).astype(int)
        # 情况2：标签是整数（如2/3）→ 强制映射到0/1（超出范围的设为0）
        else:
            self.merged_df['label'] = np.where(self.merged_df['label'] == 1, 1, 0)

        # 验证标签是否合法（可选，用于调试）
        unique_labels = self.merged_df['label'].unique()
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"{self.split}集标签非法！当前标签：{unique_labels}，必须是0或1")
        print(f"{self.split}集标签分布：0→{sum(self.merged_df['label'] == 0)}个，1→{sum(self.merged_df['label'] == 1)}个")

    def __len__(self):
        return len(self.merged_df)

    def __getitem__(self, idx):
        row = self.merged_df.iloc[idx]
        sample_id = str(row["ID"]).strip()

        # 加载三类图像（返回键为 'image'）
        images = []
        for folder, ext in self.image_folders:
            img_path = os.path.join(self.base_dir, folder, f"{sample_id}{ext}")
            img = Image.open(img_path).convert('RGB')  # 统一转为3通道（避免灰度图维度问题）
            images.append(self.transform(img))
        images = torch.stack(images, dim=0)  # 形状：(3, 3, 224, 224)（视图数, 通道数, 高, 宽）

        # 加载临床特征（确保是数值类型）
        tumor_marker = torch.tensor([row["CA199"], row["tPSA"], row["fPSA"], row["cPSA"]], dtype=torch.float32)
        other_clinical = torch.tensor([row["age"]], dtype=torch.float32)
        img_fea = torch.zeros(2048, dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.long)

        return {
            "image": images,
            "img_fea": img_fea,
            "tumor_marker": tumor_marker,
            "other_clinical": other_clinical,
            "label": label,
            "id": sample_id
        }


def get_dataloader(data_dir, excel_path, split, batch_size=8, shuffle=True, num_workers=0):
    dataset = ProstateDataset(data_dir, excel_path, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # 自动适配CUDA（减少数据传输耗时）
        drop_last=False  # 不丢弃最后一个不完整批次（避免样本浪费）
    )