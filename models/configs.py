import torch

class Config:
    def __init__(self):
        # 图像配置
        self.num_image_views = 3  # 3类图像输入：超声图像、边缘马赛克、距离图像
        self.image_size = (224, 224)  # 统一图像尺寸
        self.in_channels = 3  # 图像通道数（灰度图可扩展为3通道）
        self.patch_size = 16  # 图像补丁大小
        self.hidden_size = 768  # Transformer隐藏层维度
        self.tumor_marker_len = 4  # 肿瘤标志物数量
        self.tumor_marker_cols = ['CA199', 'cPSA', 'fPSA', 'tPSA']  # 新增：肿瘤标志物列名
        self.non_marker_clinical_len = 1  # 非标志物临床特征数量
        self.clinical_cols = ['age']  #非标志物临床特征列名
        # Transformer配置
        self.transformer = {
            "num_layers": 16,  # 总层数
            "num_heads": 12,   # 注意力头数
            "cross_attention_layers": 4  # 跨模态注意力层数
        }
        
        # 特征维度配置
        self.image_feature_length = 2048  # 图像特征向量维度
        self.tumor_marker_len = 4  # 肿瘤标志物数量（CA199、cPSA、fPSA、tPSA）
        self.non_marker_clinical_len = 1  # 非标志物临床特征（年龄）
        self.clinical_feature_length = self.tumor_marker_len + self.non_marker_clinical_len
        
        # 输出配置
        self.num_classes = 1 # 二分类
        self.dropout_rate = 0.1  # dropout比率

# 创建配置实例
config = Config()
    