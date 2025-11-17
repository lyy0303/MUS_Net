import torch
import torch.nn as nn
import torch.nn.functional as F
from .configs import config

class Attention(nn.Module):
    def __init__(self, is_cross_attention=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.transformer["num_heads"]
        self.head_size = self.hidden_size // self.num_heads
        self.is_cross_attention = is_cross_attention
        
        # 查询、键、值投影
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 输出投影
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

    def transpose_for_scores(self, x):
        # 重塑为多头注意力格式
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_size)

    def forward(self, hidden_states, cross_attention_states=None):
        # 自注意力: cross_attention_states=None
        # 跨注意力: cross_attention_states为另一模态的特征
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        
        if self.is_cross_attention and cross_attention_states is not None:
            key_layer = self.transpose_for_scores(self.key(cross_attention_states))
            value_layer = self.transpose_for_scores(self.value(cross_attention_states))
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.head_size ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        # 重塑为隐藏层维度
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # 输出投影
        output = self.out_proj(context_layer)
        return output, attention_probs


class TransformerBlock(nn.Module):
    def __init__(self, is_cross_attention_block=False):
        super().__init__()
        self.config = config
        
        # 注意力层
        self.attention = Attention(is_cross_attention=is_cross_attention_block)
        
        # 前馈网络
        self.intermediate = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.intermediate_act_fn = nn.GELU()
        self.output = nn.Linear(config.hidden_size * 4, config.hidden_size)
        
        # 层归一化
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        
        # Dropout
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, cross_attention_states=None):
        # 注意力残差连接
        residual = hidden_states
        hidden_states = self.layernorm1(hidden_states)
        
        attention_output, attention_probs = self.attention(
            hidden_states, cross_attention_states
        )
        
        attention_output = self.dropout1(attention_output)
        hidden_states = residual + attention_output
        
        # 前馈网络残差连接
        residual = hidden_states
        hidden_states = self.layernorm2(hidden_states)
        
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.output(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, attention_probs
    