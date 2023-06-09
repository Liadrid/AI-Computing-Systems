import torch.nn
from torch import nn


class ConvLayer(nn.Module):
    """
    携带对称填充的卷积层

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)  # 映射填充
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, X):
        out = self.reflection_pad(X)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    """
    参差块
    torch社区博客：http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)  # 正则化
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, X):
        pre_X = X
        out = self.relu(self.in1(self.conv1(X)))
        out = self.in2(self.conv2(out))
        out = out + pre_X
        return out


class UpSampleConvLayer(nn.Module):
    """
    反卷积防止出现卷积重叠采样和Checkerboard(棋盘）,实际上就是用插值来做超采样
    参考：https://distill.pub/2016/deconv-checkerboard/
    拥有比ConvTranspose2d更好的效果
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpSampleConvLayer, self).__init__()
        self.upsample = upsample
        reflect_padding = kernel_size // 2
        self.reflect_pad = nn.ReflectionPad2d(reflect_padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, X):
        X_in = X
        if self.upsample:
            X_in = nn.functional.interpolate(X_in, mode="nearest", scale_factor=self.upsample)
        out = self.reflect_pad(X_in)
        out = self.conv(out)
        return out


class TransformerNet(nn.Module):
    """
    2016 快速风格迁移网络
    https://arxiv.org/abs/1603.08155
    """

    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.de_conv1 = UpSampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.de_conv2 = UpSampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.de_conv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)
        return y


