import torch
import torch.nn as nn
from .configs import config

class ImageEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        
        # 补丁嵌入层（共享于所有视图）
        self.patch_embeddings = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        # 计算单个视图的补丁数量
        num_patches = (config.image_size[0] // config.patch_size) * \
                      (config.image_size[1] // config.patch_size)
        self.num_patches_per_view = num_patches
        
        # 视图级CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
        # 位置编码（包含所有视图的补丁和CLS token）
        total_patches = config.num_image_views * (num_patches + 1)  # +1是每个视图的CLS token
        self.position_embeddings = nn.Parameter(
            torch.randn(1, total_patches, config.hidden_size)
        )
        
        # 视图类型编码（区分3类不同图像）
        self.view_type_embeddings = nn.Parameter(
            torch.randn(1, config.num_image_views, config.hidden_size)
        )
        
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        # x形状: (batch_size, num_views, in_channels, height, width)
        batch_size = x.shape[0]
        view_embeddings = []
        
        for view_idx in range(config.num_image_views):
            # 提取单个视图
            img = x[:, view_idx, :, :, :]  # (batch_size, in_channels, height, width)
            
            # 补丁嵌入
            patch_embeds = self.patch_embeddings(img)  # (batch_size, hidden_size, n_patches_h, n_patches_w)
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # (batch_size, num_patches, hidden_size)
            
            # 添加视图级CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_size)
            view_feats = torch.cat((cls_tokens, patch_embeds), dim=1)  # (batch_size, num_patches+1, hidden_size)
            
            # 添加视图类型编码
            view_type_embed = self.view_type_embeddings[:, view_idx, :].unsqueeze(0)  # (1, 1, hidden_size)
            view_feats = view_feats + view_type_embed  # 广播到所有样本
            
            view_embeddings.append(view_feats)
        
        # 拼接所有视图的特征
        all_feats = torch.cat(view_embeddings, dim=1)  # (batch_size, total_patches, hidden_size)
        
        # 添加位置编码并dropout
        embeddings = all_feats + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        return embeddings


class ClinicalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        
        # 肿瘤标志物嵌入
        self.tumor_marker_embedding = nn.Sequential(
            nn.Linear(config.tumor_marker_len, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 其他临床特征嵌入
        self.other_clinical_embedding = nn.Sequential(
            nn.Linear(config.non_marker_clinical_len, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 特征类型编码（区分不同临床特征）
        self.tumor_type_embed = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.clinical_type_embed = nn.Parameter(torch.randn(1, 1, config.hidden_size))

    def forward(self, tumor_marker, other_clinical):
        # 肿瘤标志物嵌入: (batch_size, tumor_marker_len) → (batch_size, 1, hidden_size)
        tumor_emb = self.tumor_marker_embedding(tumor_marker).unsqueeze(1)
        tumor_emb = tumor_emb + self.tumor_type_embed  # 添加类型编码
        
        # 其他临床特征嵌入: (batch_size, non_marker_clinical_len) → (batch_size, 1, hidden_size)
        clinical_emb = self.other_clinical_embedding(other_clinical).unsqueeze(1)
        clinical_emb = clinical_emb + self.clinical_type_embed  # 添加类型编码
        
        # 拼接临床特征: (batch_size, 2, hidden_size)
        return torch.cat([tumor_emb, clinical_emb], dim=1)


class ImageFeatureEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(config.image_feature_length, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        # 特征类型编码
        self.type_embed = nn.Parameter(torch.randn(1, 1, config.hidden_size))

    def forward(self, x):
        # 图像特征向量嵌入: (batch_size, 2048) → (batch_size, 1, hidden_size)
        emb = self.embedding(x).unsqueeze(1)
        return emb + self.type_embed  # 添加类型编码
    