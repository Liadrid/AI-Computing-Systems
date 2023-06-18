import logging

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import utils as vutils

from attack import *

attack_method = ['fsgm', 'fgm']
epsilons = [0.05, 0.1]


class BasicTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        # 数据记录
        self.train_counter = []
        self.train_accuracy_list = []
        self.test_accuracy_list = []
        # self.test2_accuracy_list = []
        # self.test3_accuracy_list = []
        # self.test4_accuracy_list = []
        # self.test5_accuracy_list = []
        self.fsgm_is_saved = False
        self.fgm_is_saved = False

        self.attack_test_accuracy_list = {}
        for m in attack_method:
            for e in epsilons:
                s = m + str(e)
                self.attack_test_accuracy_list[s] = []

    def train(self, train_loader, test_loader):
        self.model.train()
        best_loss = 0x3f3f3f
        best_epoch = -1
        for epoch in range(self.config.num_epochs):
            running_loss = 0.0
            for idx, (images, labels) in enumerate(train_loader, 0):
                images, labels = images.to(self.device), labels.to(self.device)
                images.requires_grad = True
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                if self.config.adverse_train:
                    loss /= 2.0
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    adverse_images = fgm_attack(images, 0.05, images.grad.data)
                    adverse_outputs = self.model(adverse_images)
                    loss = self.criterion(adverse_outputs, labels) / 2.0
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            loss = running_loss / len(train_loader)
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch + 1
            logging.info(f'Epoch [{epoch + 1}/{self.config.num_epochs}], '
                         f'Loss: {loss:.4f}')

            # 记录数据

            self.train_counter.append(epoch + 1)

            train_accuracy = self.test(train_loader)
            self.train_accuracy_list.append(train_accuracy)
            test_accuracy = self.test(test_loader)
            self.test_accuracy_list.append(test_accuracy)
            # test2_accuracy = self.test_with_attack(test_loader, 0.05, 'fsgm')
            # test3_accuracy = self.test_with_attack(test_loader, 0.1, 'fsgm')
            # test4_accuracy = self.test_with_attack(test_loader, 0.05, 'fgm')
            # test5_accuracy = self.test_with_attack(test_loader, 0.1, 'fgm')

            # self.test2_accuracy_list.append(test2_accuracy)
            # self.test3_accuracy_list.append(test3_accuracy)
            # self.test4_accuracy_list.append(test4_accuracy)
            # self.test5_accuracy_list.append(test5_accuracy)
            for m in attack_method:
                for e in epsilons:
                    attack_accuracy = self.test_with_attack(test_loader, e, m)
                    s = m + str(e)
                    self.attack_test_accuracy_list[s].append(attack_accuracy)

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
        return 100 * correct / total

    def save(self):
        torch.save(self.model.state_dict(), self.config.save_path)

    def load(self):
        self.model.load_state_dict(torch.load(self.config.save_path))

    def test_with_attack(self, test_loader, _e, method_id):
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
            if method_id == 'fsgm':
                perturbed_images = fsgm_attack(images, _e, grad)
                if not self.fsgm_is_saved:
                    vutils.save_image(perturbed_images, 'image/fsgm_image.png')
                    vutils.save_image(images, 'image/fsgm_noattack_image.png')
                    self.fsgm_is_saved = True
            elif method_id == 'fgm':
                perturbed_images = fgm_attack(images, _e, grad)
                if not self.fgm_is_saved:
                    vutils.save_image(perturbed_images, 'image/fgm_image.png')
                    vutils.save_image(images, 'image/fgm_noattack_image.png')
                    self.fgm_is_saved = True
            outputs = self.model(perturbed_images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        logging.info(
            f'Test With Attack, Method: {method_id}, Epsilon: {_e}, Accuracy: {100 * correct / total:.2f}%')

        return 100 * correct / total

    def show_figure(self):
        plt.figure()
        colors = ['slategray', 'maroon', 'goldenrod', 'darkorange']
        legend = ['Train Accuracy', 'Test Accuracy(No Attack)']
        plt.plot(self.train_counter, self.train_accuracy_list, color='blue')
        plt.plot(self.train_counter, self.test_accuracy_list, color='green')
        # plt.plot(self.train_counter, self.test2_accuracy_list, color='slategray')
        # plt.plot(self.train_counter, self.test3_accuracy_list, color='maroon')
        # plt.plot(self.train_counter, self.test4_accuracy_list, color='goldenrod')
        # plt.plot(self.train_counter, self.test5_accuracy_list, color='darkorange')
        # plt.legend(['Train Accuracy', 'Test Accuracy(No Attack)', 'Epsilon:0.05, fsgm',
        #             'Epsilon:0.1, fsgm', 'Epsilon:0.05, fgm', 'Epsilon:0.1, fgm'], loc='lower right')
        color_idx = 0
        for m in attack_method:
            for e in epsilons:
                s = m + str(e)
                plt.plot(self.train_counter, self.attack_test_accuracy_list[s], colors[color_idx])
                color_idx += 1
                legend.append(f'Epsilon:{e}, {m}')
        plt.legend(legend, loc='lower right')
        plt.xlabel('number of epoches')
        plt.ylabel('accuracy')
        plt.show()
