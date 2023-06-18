import matplotlib.pyplot as plt

from naive_trainer import *
from configs import Configs
from PIL import Image
from torchvision import transforms

import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def postprocess(img):
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return transforms.ToPILImage()(img.permute(2, 0, 1))


if __name__ == '__main__':
    configs = Configs()
    naive_trainer = NaiveTrainer(configs)
    content_image = Image.open('images/content_imgs/amber.jpg')
    style_image = Image.open('images/style_images/mosaic.jpg')
    content_X, content_Y = naive_trainer.extract_content(content_image)
    _, style_Y = naive_trainer.extract_style(style_image)
    output = naive_trainer.train(content_X, content_Y, style_Y)
    output = postprocess(output)
    output.show()
    output.save("images/syn_images/output.png")
