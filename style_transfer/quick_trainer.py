import imageio
import matplotlib.pyplot as plt
import torch.optim
from PIL import ImageSequence
from torchvision.models import VGG16_Weights

from basic_trainer import *
from torchvision import models, datasets
from torchvision import transforms

from configs import Configs
from quick_transfer_net import TransformerNet
from CocoDataset import CocoDataset
from style_transfer.vgg_net import Vgg16
from utils import *
from utils import gram_matrix
from torch.utils.data import DataLoader

rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])


class QuickTrainer(BasicTrainer):
    logging.basicConfig(filename='fst_train.log', format='FST:%(asctime)s %(message)s', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)

    """
    快速风格迁移中的风格转换网络训练器
    """

    def __init__(self, config, style_img):
        super().__init__(config)
        self.model = TransformerNet().to(self.config.device)
        self.style_img = style_img
        # 使用vgg16
        self.loss_net = Vgg16().to(self.config.device)
        self.transform = transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std),
            transforms.Lambda(lambda x: x.mul(255))])

    def extract_style(self, style_img):
        # 处理style图片
        style_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.config.img_size),
            transforms.Lambda(lambda x: x.mul(255))
        ])
        style = style_transform(style_img)
        style = style.repeat(self.config.batch_size, 1, 1, 1).to(self.config.device)
        features_style = self.loss_net(normalize_batch(style))
        gram_style = [gram_matrix(y) for y in features_style]

        return gram_style

    def stylize(self, img):
        # 对给定图像执行风格迁移
        self.model.eval()
        with torch.no_grad():
            img = self.model(self.transform(img).to(self.config.device).unsqueeze(0))
        return img

    def stylize_gif(self, addr):
        if addr.endswith(".gif"):
            # gif = imageio.imread(uri=args.content_image, format=".gif")
            gif = Image.open(addr)
            output_frames = []
            with torch.no_grad():
                state_dict = torch.load(self.config.model_save_path)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()
                for frame in ImageSequence.Iterator(gif):
                    content_image = self.transform(frame.convert("RGB"))
                    input_tensor = content_image.unsqueeze(0).to(self.device)
                    output_tensor = self.model(input_tensor).cpu()
                    output_frame = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
                    output_frames.append(np.uint8(output_frame))
            imageio.mimsave(self.config.output, output_frames, loop=0)

    def train(self, data_loader):
        gram_style = self.extract_style(self.style_img)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        loss_func = nn.MSELoss()

        logging.info(f'Training start.')

        for epoch in range(self.config.num_epochs):
            self.model.train()
            agg_content_loss = 0
            agg_style_loss = 0
            count = 0

            for batch_id, x in enumerate(data_loader):
                n_batch = len(x)
                count += n_batch
                optimizer.zero_grad()

                x = x.to(self.config.device)
                y = self.model(x)

                # 计算content loss
                x = utils.normalize_batch(x)
                y = utils.normalize_batch(y)

                features_x = self.loss_net(x)
                features_y = self.loss_net(y)

                content_l = self.config.content_weight * loss_func(features_y.relu2_2, features_x.relu2_2)

                # 计算style loss
                style_l = 0
                for ft_y, gm_s in zip(features_y, gram_style):
                    gm_y = gram_matrix(ft_y)
                    style_l += loss_func(gm_y, gm_s)
                    style_l *= self.config.style_weight

                total_loss = content_l + style_l
                total_loss.backward()
                optimizer.step()

                agg_content_loss += content_l.item()
                agg_style_loss += style_l.item()

                if (batch_id + 1) % 500 == 0:
                    logging.info(f'Epoch[{epoch + 1}/{self.config.num_epochs}], '
                                 f'content loss: {agg_content_loss / (batch_id + 1):.2f}, '
                                 f'style loss: {agg_style_loss / (batch_id + 1):.2f}, '
                                 f'total loss: {(agg_content_loss + agg_style_loss) / (batch_id + 1):.2f}')

            torch.save(self.model.state_dict(), self.config.model_save_path)


config = Configs()
transform = transforms.Compose([
    transforms.Resize(Configs().img_size),
    transforms.ToTensor(),
    transforms.Resize(config.img_size),
    transforms.Lambda(lambda x: x.mul(255))
])
dataset = CocoDataset(root_dir='./dataset', transform=transform)
data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
style_image = load_image('images/style_images/mosaic.jpg')
trainer = QuickTrainer(config, style_image)
trainer.train(data_loader)

# trainer.model.load_state_dict(torch.load(config.model_save_path))
img = Image.open('images/content_imgs/amber.jpg')
img = trainer.stylize(img).squeeze(0)
img = transforms.ToPILImage()(img)
plt.imshow(img)
plt.axis('off')
plt.show()
