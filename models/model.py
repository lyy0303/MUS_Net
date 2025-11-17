import torch
import torch.nn as nn
from models.configs import (config)
from models.embed import ImageEmbedding, ClinicalEmbedding, ImageFeatureEmbedding
from models.block import TransformerBlock

class LLNM_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.image_embedding = ImageEmbedding()
        self.clinical_embedding = ClinicalEmbedding()
        self.image_feature_embedding = ImageFeatureEmbedding()
        
        # Transformer层
        self.cross_attention_layers = nn.ModuleList([
            TransformerBlock(is_cross_attention_block=True)
            for _ in range(config.transformer["cross_attention_layers"])
        ])
        
        self.self_attention_layers = nn.ModuleList([
            TransformerBlock(is_cross_attention_block=False)
            for _ in range(config.transformer["num_layers"] - config.transformer["cross_attention_layers"])
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.num_classes)
        )
        
        # 损失函数
        if config.num_classes == 1:
            self.loss_fct = nn.BCEWithLogitsLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, x, img_fea, tumor_marker, other_clinical, labels=None):
        # 1. 特征嵌入
        # 图像嵌入: (batch_size, num_views, channels, h, w) → (batch_size, seq_len_img, hidden_size)
        image_emb = self.image_embedding(x)
        
        # 临床特征嵌入: 肿瘤标志物 + 其他临床特征 → (batch_size, 2, hidden_size)
        clinical_emb = self.clinical_embedding(tumor_marker, other_clinical)
        
        # 图像特征向量嵌入: (batch_size, 2048) → (batch_size, 1, hidden_size)
        img_fea_emb = self.image_feature_embedding(img_fea)
        
        # 合并非图像特征作为跨注意力的"文本"端
        text_emb = torch.cat([clinical_emb, img_fea_emb], dim=1)  # (batch_size, 3, hidden_size)
        
        # 2. Transformer编码
        hidden_states = image_emb
        cross_attention_weights = []
        self_attention_weights = []
        
        # 前4层: 跨模态注意力（图像特征 ←→ 临床+图像特征向量）
        for layer in self.cross_attention_layers:
            hidden_states, attn_probs = layer(hidden_states, text_emb)
            cross_attention_weights.append(attn_probs)
            
            # 同时更新文本端特征
            text_emb, _ = layer(text_emb, hidden_states)
        
        # 第5层: 拼接所有特征
        hidden_states = torch.cat([hidden_states, text_emb], dim=1)
        
        # 剩余层: 自注意力
        for layer in self.self_attention_layers:
            hidden_states, attn_probs = layer(hidden_states)
            self_attention_weights.append(attn_probs)
        
        # 3. 分类
        # 取第一个CLS token（来自第一个图像视图的CLS）
        cls_token = hidden_states[:, 0, :]
        logits = self.classifier(cls_token)
        
        # 计算损失
        loss = None
        if labels is not None:
            if self.config.num_classes ==1:
                loss = self.loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                loss = self.loss_fct(logits, labels)
        
        return {
            "logits": logits,
            "loss": loss,
            "cross_attention_weights": cross_attention_weights,
            "self_attention_weights": self_attention_weights
        }
    