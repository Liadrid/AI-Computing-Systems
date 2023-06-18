import logging

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
import utils


class BasicTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None

        self.criterion = nn.CrossEntropyLoss()
        self.content_optimizer = None
        self.transform_optimizer = None

        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # 图片变换
        self.transform = transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.CenterCrop(self.config.img_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255))
        ])

    def train(self, **kwargs):
        pass

    def stylize(self):
        pass
