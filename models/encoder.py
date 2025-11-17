import torch
import torch.nn as nn
from .configs import config
from .block import TransformerBlock

class Encoder(nn.Module):
    """Transformer编码器"""
    def __init__(self):
        super().__init__()
        # 创建Transformer层
        self.layers = nn.ModuleList()
        # 前几层使用跨模态注意力
        for _ in range(config.transformer["cross_attention_layers"]):
            self.layers.append(TransformerBlock(is_cross_attention=True))
        # 剩余层使用自注意力
        remaining_layers = config.transformer["num_layers"] - config.transformer["cross_attention_layers"]
        for _ in range(remaining_layers):
            self.layers.append(TransformerBlock(is_cross_attention=False))

    def forward(self, hidden_states, text_hidden):
        all_self_weights = []
        all_cross_weights = []
        
        # 处理前几层的跨模态注意力
        for i in range(config.transformer["cross_attention_layers"]):
            hidden_states, text_hidden, (self_weights, cross_weights) = self.layers[i](hidden_states, text_hidden)
            all_self_weights.append(self_weights)
            all_cross_weights.append(cross_weights)
        
        # 拼接图像和文本特征
        hidden_states = torch.cat((hidden_states, text_hidden), dim=1)
        
        # 处理剩余层的自注意力
        for i in range(config.transformer["cross_attention_layers"], config.transformer["num_layers"]):
            hidden_states, self_weights = self.layers[i](hidden_states)
            all_self_weights.append(self_weights)
        
        return hidden_states, all_self_weights, all_cross_weights
