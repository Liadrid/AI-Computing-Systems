# 参考 https://github.com/pytorch/examples/blob/0.4/fast_neural_style/neural_style/utils.py
# 参考 https://zh-v2.d2l.ai/chapter_computer-vision/neural-style.html#id11
import torch
from PIL import Image
from matplotlib_inline import backend_inline
from matplotlib import pyplot as plt
from IPython import display


# 加载
def load_image(file_name, size=None, scale=None):
    img = Image.open(file_name)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)  # ANTIALIAS表示高质量（还有双线性等对图像低帧率处理的方式）
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


# 保存
def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


# 内容损失
def content_loss(Y_hat, Y):
    """
     (Gatys et al., 2016) 描述的风格迁移函数
    :param Y_hat: 神经网络的输出
    :param Y: 原始图像
    :return:
    """
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。
    return torch.square(Y_hat - Y.detach()).mean()


# Gram 矩阵
def gram_matrix(X):
    """
    计算格拉姆矩阵，目的是将(c,h,w)的图像拉平成（c, h * w)并用举证描述c个长为h*w的向量中
    x_i和x_j的内积,表达每个通道之间的相关性
    :param X: 输入图像
    :return:
    """
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    return torch.matmul(X, X.T) / (num_channels * n)


# 迁移损失
def style_loss(Y_hat, gram_Y):
    """
    计算格拉姆矩阵，求均值
    :param Y_hat:
    :param gram_Y:
    :return:
    """
    # gram = gram_matrix(Y_hat)
    # print(gram.shape, gram_Y.shape)
    return torch.square(gram_matrix(Y_hat) - gram_Y.detach()).mean()


# 全变分损失，原始风格迁移中使用,用于消除噪点
def tv_loss(Y_hat):
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


# 标准化BN
def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


class Animator:  # @save
    """在动画中绘制数据,参考d2l官方文档"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        backend_inline.set_matplotlib_formats('svg')  # 使用svg的格式显示
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 设置横纵坐标轴
        self.config_axes = lambda: self.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    @staticmethod
    def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib.

        Defined in :numref:`sec_calculus`"""
        axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
        axes.set_xscale(xscale), axes.set_yscale(yscale)
        axes.set_xlim(xlim), axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        # 保存
        self.fig.savefig("images/output/output.png")
