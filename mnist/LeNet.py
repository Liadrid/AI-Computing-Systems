import logging

import torch
import torch.nn as nn
import torch.optim as optim

from attack import *


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


class LeNetTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = LeNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

    def train(self, train_loader):
        self.model.train()
        best_loss = 0x3f3f3f
        best_epoch = -1
        for epoch in range(self.config.num_epochs):
            running_loss = 0.0
            for idx, (images, labels) in enumerate(train_loader, 0):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            loss = running_loss / len(train_loader)
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch + 1
            logging.info(f'Epoch [{epoch + 1}/{self.config.num_epochs}], '
                         f'Loss: {loss:.4f}')
        logging.info(f'Training completed. Best loss is {best_loss}, in epoch {best_epoch}')

    def test(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        logging.info(f'Test Without Attack, Accuracy: {100 * correct / total:.2f}%')

    def save(self):
        torch.save(self.model.state_dict(), self.config.save_path)

    def test_with_attack(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            images.requires_grad = True
            outputs = self.model(images)

            perturbed_images = images
            loss = self.criterion(outputs, labels)
            self.model.zero_grad()
            loss.backward()
            grad = images.grad.data
            if self.config.attack == 'fsgm':
                perturbed_images = fsgm_attack(images, self.config.epsilon, grad)
            elif self.config.attack == 'fgm':
                perturbed_images = fgm_attack(images, self.config.epsilon, grad)
            outputs = self.model(perturbed_images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        logging.info(
            f'Test With Attack, Method: {self.config.attack}, Epsilon: {self.config.epsilon}, Accuracy: {100 * correct / total:.2f}%')
