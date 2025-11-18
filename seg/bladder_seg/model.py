import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# 基础卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# UNet++ 编码器
class UNetPlusPlusEncoder(nn.Module):
    def __init__(self, in_channels=3, use_pretrained=True):
        super(UNetPlusPlusEncoder, self).__init__()

        # 使用预训练的ResNet34作为编码器
        resnet = models.resnet34(pretrained=use_pretrained)

        # 修改第一层卷积以适应不同的输入通道数
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = resnet.conv1

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 输出通道: 64
        self.layer2 = resnet.layer2  # 输出通道: 128
        self.layer3 = resnet.layer3  # 输出通道: 256
        self.layer4 = resnet.layer4  # 输出通道: 512

    def forward(self, x):
        # 初始卷积
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0_pool = self.maxpool(x0)

        # 各层特征
        x1 = self.layer1(x0_pool)  # 1/4 尺度, 64通道
        x2 = self.layer2(x1)  # 1/8 尺度, 128通道
        x3 = self.layer3(x2)  # 1/16 尺度, 256通道
        x4 = self.layer4(x3)  # 1/32 尺度, 512通道

        # 返回所有特征图，包括初始卷积结果
        return x0, x1, x2, x3, x4


# UNet++中的上采样融合模块
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


# UNet++解码器（修正通道数）
class UNetPlusPlusDecoder(nn.Module):
    def __init__(self, num_classes):
        super(UNetPlusPlusDecoder, self).__init__()

        # 上采样层 - 修正通道数
        self.up4 = UpConv(512, 256)  # 512 -> 256
        self.up3 = UpConv(256, 128)  # 256 -> 128
        self.up2 = UpConv(128, 64)  # 128 -> 64
        self.up1 = UpConv(64, 64)  # 64 -> 64

        # 卷积块用于特征融合
        self.conv4 = ConvBlock(512, 256)  # 256(上采样) + 256(跳跃连接) = 512
        self.conv3 = ConvBlock(256, 128)  # 128 + 128 = 256
        self.conv2 = ConvBlock(128, 64)  # 64 + 64 = 128
        self.conv1 = ConvBlock(128, 64)  # 64 + 64 = 128

        # 最终卷积
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x0, x1, x2, x3, x4):
        """
        x0: 初始卷积输出 (64通道)
        x1: layer1输出 (64通道)
        x2: layer2输出 (128通道)
        x3: layer3输出 (256通道)
        x4: layer4输出 (512通道)
        """

        # 解码路径
        d4 = self.up4(x4)  # 512->256
        d4 = torch.cat([d4, x3], dim=1)  # 256 + 256 = 512
        d4 = self.conv4(d4)  # 512->256

        d3 = self.up3(d4)  # 256->128
        d3 = torch.cat([d3, x2], dim=1)  # 128 + 128 = 256
        d3 = self.conv3(d3)  # 256->128

        d2 = self.up2(d3)  # 128->64
        d2 = torch.cat([d2, x1], dim=1)  # 64 + 64 = 128
        d2 = self.conv2(d2)  # 128->64

        d1 = self.up1(d2)  # 64->64
        d1 = torch.cat([d1, x0], dim=1)  # 64 + 64 = 128
        d1 = self.conv1(d1)  # 128->64

        # 最终输出
        output = self.final_conv(d1)
        return output


# 纯UNet++分割模型
class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes, in_channels=3, use_pretrained=True):
        super(UNetPlusPlus, self).__init__()
        self.encoder = UNetPlusPlusEncoder(in_channels, use_pretrained)
        self.decoder = UNetPlusPlusDecoder(num_classes)

    def forward(self, x):
        # 获取输入尺寸
        input_size = x.shape[2:]

        # 编码
        x0, x1, x2, x3, x4 = self.encoder(x)

        # 解码
        output = self.decoder(x0, x1, x2, x3, x4)

        # 调整输出尺寸
        if output.shape[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)

        return output


# 简化版本（如果上面的仍然有问题）
class SimpleUNetPlusPlus(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(SimpleUNetPlusPlus, self).__init__()

        # 编码器
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 解码器
        self.up4 = self._up_block(512, 256)
        self.up3 = self._up_block(512, 128)  # 256*2 = 512
        self.up2 = self._up_block(256, 64)  # 128*2 = 256
        self.up1 = self._up_block(128, 64)  # 64*2 = 128

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码路径
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # 解码路径
        d4 = self.up4(e4)
        d3 = self.up3(torch.cat([d4, e3], dim=1))
        d2 = self.up2(torch.cat([d3, e2], dim=1))
        d1 = self.up1(torch.cat([d2, e1], dim=1))

        return self.final_conv(d1)


# 测试模型
if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建纯UNet++模型实例
    model = UNetPlusPlus(num_classes=2).to(device)

    # 或者使用简化版本
    # model = SimpleUNetPlusPlus(num_classes=2).to(device)

    # 创建测试输入
    test_input = torch.randn(2, 3, 256, 256).to(device)

    # 前向传播
    output = model(test_input)

    # 打印输入输出形状
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数: {total_params / 1e6:.2f} M")