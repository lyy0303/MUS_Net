import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


# 基础卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()  # YOLO中常用的激活函数

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# CSP瓶颈块 - YOLOv12中使用的高效模块
class CSPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super(CSPBottleneck, self).__init__()
        self.split_channels = out_channels // 2
        self.conv1 = ConvBlock(in_channels, self.split_channels, 1, 1, 0)
        self.conv2 = ConvBlock(in_channels, self.split_channels, 1, 1, 0)

        self.bottlenecks = nn.Sequential(
            *[ConvBlock(self.split_channels, self.split_channels) for _ in range(n)]
        )

        self.conv3 = ConvBlock(2 * self.split_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = self.bottlenecks(x1)
        return self.conv3(torch.cat([x1, x2], dim=1))


# YOLOv12特征提取器
class YOLOv12Backbone(nn.Module):
    def __init__(self, in_channels=3):
        super(YOLOv12Backbone, self).__init__()
        # 初始卷积
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 64, 3, 2, 1),
            ConvBlock(64, 128, 3, 2, 1),
            CSPBottleneck(128, 128)
        )

        # 下采样和特征提取
        self.layer1 = nn.Sequential(
            ConvBlock(128, 256, 3, 2, 1),
            CSPBottleneck(256, 256, n=2)
        )

        self.layer2 = nn.Sequential(
            ConvBlock(256, 512, 3, 2, 1),
            CSPBottleneck(512, 512, n=8)
        )

        self.layer3 = nn.Sequential(
            ConvBlock(512, 1024, 3, 2, 1),
            CSPBottleneck(1024, 1024, n=8)
        )

        self.layer4 = nn.Sequential(
            ConvBlock(1024, 2048, 3, 2, 1),
            CSPBottleneck(2048, 2048, n=4)
        )

    def forward(self, x):
        # 输出不同尺度的特征图用于Unet++
        x = self.stem(x)  # 1/4 尺度
        x1 = self.layer1(x)  # 1/8 尺度
        x2 = self.layer2(x1)  # 1/16 尺度
        x3 = self.layer3(x2)  # 1/32 尺度
        x4 = self.layer4(x3)  # 1/64 尺度
        return x, x1, x2, x3, x4


# Unet++中的上采样融合模块
class UpCat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpCat, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            ConvBlock(2 * out_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        # 处理尺寸不匹配的情况
        if x.shape[2:] != skip_x.shape[2:]:
            x = F.interpolate(x, size=skip_x.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)


# Unet++中的嵌套连接模块
class NestedUNetBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(NestedUNetBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, mid_channels)
        self.conv2 = ConvBlock(mid_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


# Unet++解码器
class UNetPlusPlusDecoder(nn.Module):
    def __init__(self, num_classes):
        super(UNetPlusPlusDecoder, self).__init__()
        # 从YOLOv12 backbone的输出通道数开始
        self.up5 = UpCat(2048, 1024)
        self.up4 = UpCat(1024, 512)
        self.up3 = UpCat(512, 256)
        self.up2 = UpCat(256, 128)
        self.up1 = UpCat(128, 64)

        # 嵌套连接
        self.nest1 = NestedUNetBlock(1024 + 1024, 1024, 1024)
        self.nest2 = NestedUNetBlock(512 + 512, 512, 512)
        self.nest3 = NestedUNetBlock(256 + 256, 256, 256)
        self.nest4 = NestedUNetBlock(128 + 128, 128, 128)

        # 最终卷积以获得分割结果
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x0, x1, x2, x3, x4):
        # 上采样并与跳跃连接融合
        x5 = self.up5(x4, x3)
        x5 = self.nest1(torch.cat([x5, x3], dim=1))

        x6 = self.up4(x5, x2)
        x6 = self.nest2(torch.cat([x6, x2], dim=1))

        x7 = self.up3(x6, x1)
        x7 = self.nest3(torch.cat([x7, x1], dim=1))

        x8 = self.up2(x7, x0)
        x8 = self.nest4(torch.cat([x8, x0], dim=1))

        # 最终上采样到输入图像尺寸
        output = self.up1(x8, torch.zeros_like(x8)[:, :64, :, :])  # 使用零张量作为最低层跳跃连接
        output = self.final_conv(output)

        return output


# YOLOv12 + UNet++ 分割模型
class YOLOv12UNetPlusPlus(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(YOLOv12UNetPlusPlus, self).__init__()
        self.backbone = YOLOv12Backbone(in_channels)
        self.decoder = UNetPlusPlusDecoder(num_classes)

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 获取输入图像尺寸用于最终调整
        input_size = x.shape[2:]

        # 特征提取
        x0, x1, x2, x3, x4 = self.backbone(x)

        # 解码获取分割结果
        output = self.decoder(x0, x1, x2, x3, x4)

        # 调整输出尺寸与输入一致
        if output.shape[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)

        return output

    def _initialize_weights(self):
        # 初始化卷积层权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


# 测试模型
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型实例，假设分割10个类别
    model = YOLOv12UNetPlusPlus(num_classes=10).to(device)

    # 创建测试输入 (batch_size=2, channels=3, height=640, width=640)
    test_input = torch.randn(2, 3, 640, 640).to(device)

    # 前向传播
    output = model(test_input)

    # 打印输入输出形状
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params / 1e6:.2f} M")
