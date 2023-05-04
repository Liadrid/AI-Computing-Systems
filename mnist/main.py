import torchvision
import torchvision.transforms as transforms

from Configs import Configs
from LeNet import *

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

train_set = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

config = Configs()
config.parse()
trainer = LeNetTrainer(config)

if __name__ == '__main__':
    trainer.train(train_loader)
    trainer.test(test_loader)
    trainer.save()
    trainer.test_with_attack(test_loader)

