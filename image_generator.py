import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from sklearn.model_selection import train_test_split
import argparse
from typing import Tuple


# 随机马赛克处理类（保留核心功能，移除边缘提取相关代码）
class RandomMosaic:
    def __init__(
            self,
            block_size: Tuple[int, int] = (16, 16),  # 马赛克块大小（高×宽）
            mosaic_ratio: float = 0.3,  # 结节区域中进行马赛克处理的比例（0~1）
            扰动_type: str = "shuffle",  # 扰动类型：shuffle（像素打乱）/ swap（块交换）/ noise（噪声）
            noise_intensity: float = 0.1  # 噪声强度（仅当扰动_type为noise时生效，0~0.5）
    ):
        """随机马赛克方法初始化"""
        self.block_h, self.block_w = block_size
        self.mosaic_ratio = max(0.0, min(1.0, mosaic_ratio))  # 裁剪到0~1范围
        self.扰动_type = 扰动_type.lower()
        self.noise_intensity = max(0.0, min(0.5, noise_intensity))

        # 验证扰动类型合法性
        if self.扰动_type not in ["shuffle", "swap", "noise"]:
            raise ValueError("扰动_type仅支持'shuffle'/'swap'/'noise'")

    def _get_nodule_blocks(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, list]:
        """从图像中提取结节区域的所有马赛克块，并记录块的坐标"""
        H, W = image.shape
        block_coords = []  # 存储块的坐标（y起始，y结束，x起始，x结束）
        nodule_blocks = []  # 存储结节区域的块

        # 遍历图像，按块大小划分
        for y in range(0, H, self.block_h):
            y1 = min(y + self.block_h, H)
            for x in range(0, W, self.block_w):
                x1 = min(x + self.block_w, W)

                # 若块中结节像素占比>50%，视为结节块
                block_mask = mask[y:y1, x:x1]
                if np.sum(block_mask) / (self.block_h * self.block_w) > 0.5:
                    block = image[y:y1, x:x1].copy()
                    nodule_blocks.append(block)
                    block_coords.append((y, y1, x, x1))

        return np.array(nodule_blocks), block_coords

    def _disturb_block(self, block: np.ndarray) -> np.ndarray:
        """对单个块进行扰动处理（根据扰动类型）"""
        block = block.copy()
        H, W = block.shape

        if self.扰动_type == "shuffle":
            # 像素打乱：将块的像素展平后随机重排
            flat_block = block.flatten()
            np.random.shuffle(flat_block)
            return flat_block.reshape(H, W)

        elif self.扰动_type == "swap":
            # 块内子区域交换：将块分为4个子块，随机交换2个
            sub_h, sub_w = H // 2, W // 2
            sub_blocks = [
                block[0:sub_h, 0:sub_w],
                block[0:sub_h, sub_w:W],
                block[sub_h:H, 0:sub_w],
                block[sub_h:H, sub_w:W]
            ]
            idx1, idx2 = random.sample(range(4), 2)
            sub_blocks[idx1], sub_blocks[idx2] = sub_blocks[idx2], sub_blocks[idx1]
            return np.vstack([
                np.hstack([sub_blocks[0], sub_blocks[1]]),
                np.hstack([sub_blocks[2], sub_blocks[3]])
            ])

        elif self.扰动_type == "noise":
            # 添加高斯噪声：噪声强度基于图像像素范围
            noise = np.random.normal(0, 255 * self.noise_intensity, (H, W))
            noisy_block = block + noise
            return np.clip(noisy_block, 0, 255).astype(np.uint8)

    def apply(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """对超声图像应用随机马赛克处理（核心方法）"""
        # 步骤1：提取结节区域的块和坐标
        nodule_blocks, block_coords = self._get_nodule_blocks(image, mask)
        if len(nodule_blocks) == 0:
            return image  # 无结节区域，直接返回原图

        # 步骤2：选择需要扰动的块（按mosaic_ratio随机选择）
        n_total = len(nodule_blocks)
        n_mosaic = int(n_total * self.mosaic_ratio)
        mosaic_indices = random.sample(range(n_total), n_mosaic)

        # 步骤3：对选中的块进行扰动，并替换回原图
        result = image.copy()
        for idx in mosaic_indices:
            block = nodule_blocks[idx]
            disturbed_block = self._disturb_block(block)
            y0, y1, x0, x1 = block_coords[idx]
            result[y0:y1, x0:x1] = disturbed_block

        return result


# 设置随机种子确保可复现性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def load_model(model_path, num_classes, device):
    """加载分割模型"""
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 假设模型定义在model.py中，根据实际情况修改
    from seg.bladder_seg.model import UNetPlusPlus  # 替换为实际的模型类名
    model = UNetPlusPlus(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, img_size, device):
    """预处理图像用于模型输入"""
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    augmented = transform(image=image_np)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    return image_tensor, image_np.shape[:2]  # 返回处理后的张量和原始尺寸


def get_segmentation_mask(model, image_tensor, original_size, device):
    """使用模型获取分割掩码"""
    with torch.no_grad():
        output = model(image_tensor)
        output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=True)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return mask


def extract_prostate_roi(original_image, prostate_mask):
    """
    提取前列腺区域，使用最小外接矩形，并resize到224x224
    """
    # 确保掩码为二值图像
    prostate_binary = (prostate_mask > 0).astype(np.uint8)

    # 找到前列腺区域的轮廓
    contours, _ = cv2.findContours(prostate_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None  # 没有找到轮廓

    # 获取最大轮廓（假设最大的是前列腺）
    max_contour = max(contours, key=cv2.contourArea)

    # 计算最小外接矩形
    x, y, w, h = cv2.boundingRect(max_contour)

    # 提取ROI
    roi = original_image[y:y + h, x:x + w]
    roi_mask = prostate_binary[y:y + h, x:x + w]

    # 调整为正方形（保持比例）
    max_dim = max(w, h)
    square_roi = np.zeros((max_dim, max_dim, 3), dtype=np.uint8) if len(roi.shape) == 3 else np.zeros(
        (max_dim, max_dim), dtype=np.uint8)
    square_mask = np.zeros((max_dim, max_dim), dtype=np.uint8)

    # 计算放置位置（居中）
    x_offset = (max_dim - w) // 2
    y_offset = (max_dim - h) // 2
    square_roi[y_offset:y_offset + h, x_offset:x_offset + w] = roi
    square_mask[y_offset:y_offset + h, x_offset:x_offset + w] = roi_mask

    # Resize到224x224
    resized_roi = cv2.resize(square_roi, (224, 224), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(square_mask, (224, 224), interpolation=cv2.INTER_NEAREST)

    return resized_roi, resized_mask


def calculate_distance_map(prostate_mask, bladder_mask):
    """计算前列腺与膀胱的距离图像"""
    # 确保掩码为二值图像
    prostate_binary = (prostate_mask > 0).astype(np.uint8)
    bladder_binary = (bladder_mask > 0).astype(np.uint8)

    # 计算距离变换
    prostate_dist = cv2.distanceTransform(1 - prostate_binary, cv2.DIST_L2, 5)
    bladder_dist = cv2.distanceTransform(1 - bladder_binary, cv2.DIST_L2, 5)

    # 计算前列腺到膀胱的距离图
    distance_map = np.zeros_like(prostate_dist)
    mask = (prostate_binary > 0) | (bladder_binary > 0)
    distance_map[mask] = np.abs(prostate_dist[mask] - bladder_dist[mask])

    # 归一化到0-255
    distance_map = cv2.normalize(distance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return distance_map


def process_patient(id_dir, prostate_model, bladder_model, img_size, device):
    """处理单个病人数据，返回一个影像号的处理结果"""
    # 获取该病人所有影像号文件夹
    study_dirs = [d for d in os.listdir(id_dir) if os.path.isdir(os.path.join(id_dir, d))]
    if not study_dirs:
        return None

    # 随机选择一个影像号文件夹
    study_dir = random.choice(study_dirs)
    study_path = os.path.join(id_dir, study_dir)

    # 查找该影像号对应的图像文件
    image_files = [f for f in os.listdir(study_path) if f.lower().endswith(('.jpg', '.png'))]
    if not image_files:
        return None

    # 找到原始图像和掩码
    original_img = None
    prostate_mask_file = None
    bladder_mask_file = None

    for f in image_files:
        if '_bladdermask.png' in f:
            bladder_mask_file = os.path.join(study_path, f)
        elif '_prostatemask.png' in f:
            prostate_mask_file = os.path.join(study_path, f)
        elif not ('_bladdermask' in f or '_prostatemask' in f):
            original_img = os.path.join(study_path, f)

    # 如果缺少必要文件则跳过
    if not original_img or not prostate_mask_file or not bladder_mask_file:
        return None

    try:
        # 加载原始图像
        original_image = cv2.imread(original_img)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 加载掩码文件
        prostate_mask = np.array(Image.open(prostate_mask_file).convert('L'))
        prostate_mask = (prostate_mask > 127).astype(np.uint8)  # 转为0/1二值图

        bladder_mask = np.array(Image.open(bladder_mask_file).convert('L'))
        bladder_mask = (bladder_mask > 127).astype(np.uint8)  # 转为0/1二值图

        # 1. 提取前列腺ROI并resize到224x224
        resized_prostate, resized_mask = extract_prostate_roi(original_image, prostate_mask)
        if resized_prostate is None:
            print(f"警告：在 {id_dir}/{study_dir} 中未检测到前列腺轮廓")
            return None

        # 转换为灰度图用于马赛克处理
        prostate_gray = cv2.cvtColor(resized_prostate, cv2.COLOR_RGB2GRAY)

        # 2. 计算前列腺与膀胱的距离图像
        # 先调整膀胱掩码尺寸与原始图像匹配，再计算距离
        bladder_mask_resized = cv2.resize(bladder_mask, (original_image.shape[1], original_image.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
        distance_map = calculate_distance_map(prostate_mask, bladder_mask_resized)

        # 3. 使用RandomMosaic处理前列腺图像
        mosaic_processor = RandomMosaic(
            block_size=(16, 16),
            mosaic_ratio=0.3,
            扰动_type="shuffle",
            noise_intensity=0.1
        )

        # 应用马赛克处理
        mosaic_image = mosaic_processor.apply(prostate_gray, resized_mask)

        # 返回结果（仅保留病人ID，移除study_id）
        patient_id = os.path.basename(id_dir)
        return {
            'original': original_image,
            'resized_prostate': resized_prostate,  # 提取并resize的前列腺区域
            'distance_map': distance_map,  # 前列腺与膀胱的距离图像
            'mosaic_image': mosaic_image,  # 马赛克处理后的图像
            'id': patient_id,  # 仅病人ID
            'original_path': original_img
        }
    except Exception as e:
        print(f"处理出错 {id_dir}/{study_dir}: {str(e)}")
        return None


def save_processed_data(data, output_dir, split):
    """保存处理后的数据到指定目录（仅用ID名，添加序号避免重复）"""
    # 1. 定义保存目录（resized_prostate/distance_map/mosaic_image）
    split_dir = os.path.join(output_dir, split)
    save_dirs = {
        'resized_prostate': os.path.join(split_dir, 'resized_prostate'),
        'distance_map': os.path.join(split_dir, 'distance_map'),
        'mosaic_image': os.path.join(split_dir, 'mosaic_image')
    }
    # 创建目录（确保目录存在）
    for dir_path in save_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # 2. 核心：生成“ID+序号”的文件名，避免同ID重复覆盖
    patient_id = data['id']
    file_suffix = {
        'resized_prostate': '.jpg',
        'distance_map': '.png',
        'mosaic_image': '.png'
    }
    file_paths = {
        key: os.path.join(save_dirs[key], f"{patient_id}{file_suffix[key]}")
        for key in save_dirs.keys()
    }

    # 3. 检查同ID文件是否已存在，存在则添加序号（如ID_1.jpg, ID_2.jpg）
    for key in file_paths.keys():
        base_path = file_paths[key]
        if os.path.exists(base_path):
            # 查找当前最大序号（避免覆盖已有文件）
            idx = 1
            while os.path.exists(f"{os.path.splitext(base_path)[0]}_{idx}{file_suffix[key]}"):
                idx += 1
            # 更新为带序号的路径
            file_paths[key] = f"{os.path.splitext(base_path)[0]}_{idx}{file_suffix[key]}"

    # 4. 保存图像
    # resized_prostate（RGB转BGR适配OpenCV保存）
    cv2.imwrite(
        file_paths['resized_prostate'],
        cv2.cvtColor(data['resized_prostate'], cv2.COLOR_RGB2BGR)
    )
    # distance_map（单通道）
    cv2.imwrite(file_paths['distance_map'], data['distance_map'])
    # mosaic_image（单通道灰度图）
    cv2.imwrite(file_paths['mosaic_image'], data['mosaic_image'])


def main():
    parser = argparse.ArgumentParser(description='生成前列腺相关图像及数据集划分（无边缘特征，仅用ID命名）')
    parser.add_argument('--data-root', type=str, default="raw_dataset/IDset", help='IDest数据集根目录')
    parser.add_argument('--excel-path', type=str, default="raw_dataset/clinical_data.xlsx", help='包含病人信息的Excel文件路径')
    parser.add_argument('--prostate-model', type=str, default="seg/prostate_seg/models/best_model.pth", help='前列腺分割模型路径')
    parser.add_argument('--bladder-model', type=str, default="seg/bladder_seg/models/best_model.pth", help='膀胱分割模型路径')
    parser.add_argument('--output-dir', type=str, default='./processed_dataset', help='处理后数据集保存目录')
    parser.add_argument('--img-size', type=int, default=512, help='模型输入图像尺寸')
    parser.add_argument('--test-size', type=float, default=0.15, help='测试集比例')
    parser.add_argument('--val-size', type=float, default=0.2, help='验证集占训练集的比例')
    parser.add_argument('--num-classes', type=int, default=2, help='分割类别数量')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print("加载分割模型...")
    try:
        prostate_model = load_model(args.prostate_model, args.num_classes, device)
        bladder_model = load_model(args.bladder_model, args.num_classes, device)
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # 读取Excel文件获取病人ID列表
    print("读取病人信息...")
    try:
        df = pd.read_excel(args.excel_path)
        patient_ids = df['ID'].astype(str).tolist()
        print(f"Excel中提取到 {len(patient_ids)} 个病人ID")
    except Exception as e:
        print(f"读取Excel失败: {str(e)}")
        return

    # 检查数据根目录是否存在
    if not os.path.exists(args.data_root):
        print(f"错误：数据根目录不存在 - {args.data_root}")
        return

    # 获取数据根目录下的所有文件夹
    root_dirs = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    print(f"数据根目录 {args.data_root} 下找到 {len(root_dirs)} 个文件夹")

    # 匹配病人ID与文件夹
    all_patients = []
    missing_ids = []
    for pid in patient_ids:
        pid_dir = os.path.join(args.data_root, pid)
        if os.path.isdir(pid_dir):
            all_patients.append(pid_dir)
        else:
            missing_ids.append(pid)

    # 输出匹配结果
    print(f"成功匹配 {len(all_patients)} 个病人目录")
    if missing_ids:
        print(f"未找到的病人ID（前5个）: {missing_ids[:5]}...")

    # 空集检查
    if len(all_patients) == 0:
        print("错误：没有找到任何有效病人目录，请检查路径和ID匹配")
        return

    # 划分训练集、验证集和测试集
    print("划分数据集...")
    train_val, test = train_test_split(all_patients, test_size=args.test_size, random_state=42)
    train, val = train_test_split(train_val, test_size=args.val_size, random_state=42)

    print(f"训练集: {len(train)} 个病人, 验证集: {len(val)} 个病人, 测试集: {len(test)} 个病人")

    # 处理并保存所有数据
    for split_name, patient_dirs in [('train', train), ('val', val), ('test', test)]:
        print(f"处理{split_name}集...")
        for i, pid_dir in enumerate(patient_dirs, 1):
            print(f"进度: {i}/{len(patient_dirs)}", end='\r')
            result = process_patient(pid_dir, prostate_model, bladder_model, args.img_size, device)
            if result:
                save_processed_data(result, args.output_dir, split_name)
        print()  # 换行，避免进度覆盖

    print("所有数据处理完成!")
    print(f"处理后的数据保存在: {args.output_dir}")
    print("文件名格式：仅病人ID（同ID多图添加序号，如123.jpg, 123_1.jpg）")


if __name__ == "__main__":
    main()