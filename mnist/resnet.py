import logging

import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from attack import *
from basic_trainer import BasicTrainer
from configs import Configs


class Residual(nn.Module):  # @save
    """
    残差块
    """

    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


class ResNet(nn.Module):
    logging.basicConfig(filename='train.log', format='ResNet:%(asctime)s %(message)s', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    # 相同的方法构建2，3，4，5层残差块
    @staticmethod
    def _resnet_block(input_channels, num_channels, num_residuals,
                      first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk

    def __init__(self):
        super(ResNet, self).__init__()
        # block 1 64 7*7 步幅为2的卷积+ 3*3 步幅为2的池化
        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm2d(64), nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # b2，b3,b4,b5相当于把GoogleNet的Inception块替换成残差块
        self.b2 = nn.Sequential(*self._resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*self._resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*self._resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*self._resnet_block(256, 512, 2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Sequential(nn.Flatten(), nn.Linear(512, 10))

    def forward(self, X):
        X = self.b1(X)
        X = self.b2(X)
        X = self.b3(X)
        X = self.b4(X)
        X = self.b5(X)
        X = self.dense(self.pool(X))
        return X


class ResNetTrainer(BasicTrainer):
    def __init__(self):
        config = Configs(model="ResNet")
        config.parse()
        super().__init__(config)
        self.model = ResNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
