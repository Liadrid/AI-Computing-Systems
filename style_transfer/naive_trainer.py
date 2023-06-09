import torch.optim

from basic_trainer import *
from torchvision import models
from torchvision import transforms
from utils import content_loss, gram_matrix, style_loss, tv_loss, Animator

rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])
# 抽取风格特征的为每一个卷积层的第一层卷积
# 抽取内容特征为第四个卷积层最后一层卷积
style_layers, content_layers = [0, 5, 10, 19, 28], [25]


class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


class NaiveTrainer(BasicTrainer):
    """
    原始风格迁移的训练器，在获得已经训练好的vgg后，通过计算两个图片的content,style和tv（可选）loss进行迭代和合成
    """

    def __init__(self, config):
        super().__init__(config)
        self.content_loss = []
        self.style_loss = []
        # 参考论文原文使用的19层Vgg 16层卷积+5层池化层
        pretrained_net = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.net = nn.Sequential(*[pretrained_net.features[i] for i in
                                   range(max(content_layers + style_layers) + 1)]).to(self.device)
        self.model = nn.Sequential().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=rgb_mean, std=rgb_std)])

    # 分离各层特征
    def extract_features(self, X, c_layers, s_layers):
        contents = []
        style = []
        for i in range(len(self.net)):
            X = self.net[i](X)
            if i in s_layers:  # convi_1
                style.append(X)
            if i in c_layers:
                contents.append(X)  # conv4_4
        return contents, style

    def extract_content(self, X):
        content_X = self.transform(X).unsqueeze(0).to(self.device)
        content_Y, _ = self.extract_features(content_X, content_layers, style_layers)
        return content_X, content_Y

    def extract_style(self, X):
        style_X = self.transform(X).unsqueeze(0).to(self.device)
        _, styles_Y = self.extract_features(style_X, content_layers, style_layers)
        return style_X, styles_Y

    def compute_loss(self, X, contents_Y_hat, styles_Y_hat, content_Y, styles_Y_gram):
        contents_l = [content_loss(Y_hat, Y) * self.config.content_weight for Y_hat, Y in zip(
            contents_Y_hat, content_Y)]
        styles_l = [style_loss(Y_hat, Y) * self.config.style_weight for Y_hat, Y in zip(
            styles_Y_hat, styles_Y_gram)]
        tv_l = tv_loss(X) * self.config.tv_weight
        # 对所有损失求和
        L = sum(10 * styles_l + contents_l + [tv_l])
        return contents_l, styles_l, tv_l, L

    def train(self, *args):
        X, contents_Y, styles_Y = args
        synthesized_image = SynthesizedImage(X.shape).to(self.device)
        synthesized_image.weight.data.copy_(X)  # 初始化为X
        optimizer = torch.optim.Adam(synthesized_image.parameters(), lr=self.config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.config.lr_decay_epochs, 0.8)
        animator = Animator(xlabel='epoch', ylabel='loss', xlim=[10, self.config.num_epochs],
                            legend=['content', 'style', 'tv'], ncols=2, figsize=(7, 2.5))
        styles_Y_gram = [gram_matrix(Y) for Y in styles_Y]
        for epoch in range(self.config.num_epochs):
            optimizer.zero_grad()
            contents_Y_hat, styles_Y_hat = self.extract_features(synthesized_image(), content_layers, style_layers)
            contents_l, styles_l, tv_l, L = self.compute_loss(
                synthesized_image(), contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
            L.backward()
            optimizer.step()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                # animator.axes[1].imshow()
                animator.add(epoch + 1, [float(sum(contents_l)), float(sum(styles_l)), float(tv_l)])

        return synthesized_image()
