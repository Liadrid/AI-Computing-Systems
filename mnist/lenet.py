import logging

import torch.nn as nn
import torch.optim as optim

from attack import *
from basic_trainer import BasicTrainer
from configs import Configs


class LeNet(nn.Module):
    logging.basicConfig(filename='train.log', format='%(asctime)s %(message)s', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNetTrainer(BasicTrainer):
    def __init__(self):
        config = Configs(model="LeNet", attack='fsgm')
        config.parse()
        super().__init__(config)
        self.model = LeNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
