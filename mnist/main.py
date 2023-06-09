import torch.utils.data
import torchvision
import torchvision.transforms as transforms

from lenet import *
from resnet import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
trainer = LeNetTrainer()
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5,), (0.5,)),
#      transforms.Resize((96, 96))])
# trainer = ResNetTrainer()

train_set = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, )
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
if __name__ == '__main__':
    trainer.train(train_loader, test_loader)
    trainer.show_figure()
