import argparse


class Configs:
    def __init__(self, model='fast_neural'):
        self.lr = 0.001
        self.num_epochs = 50
        self.lr_decay_epochs = 50
        self.batch_size = 4
        self.device = 'cuda'
        self.seed = 1
        self.save_path = f'./{model}/out.png'
        self.model_save_path = f'./{model}/{model}.pth'
        self.model = model
        self.dataset_pth = "./dataset"
        self.output = "./images/syn_images/output.gif"

        # 总loss： 风格损失1000：content损失1：全变分损失10
        self.style_weight = 100
        self.content_weight = 100000
        self.tv_weight = 10
        self.img_size = (256, 256)

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-num_epochs', type=int)
        parser.add_argument('-device', type=str)
        parser.add_argument('-lr', type=float)
        parser.add_argument('-seed', type=int)
        parser.add_argument('-save_path', type=str)
        parser.add_argument('-model', type=str, choices=['naive_neural', 'fast_neural'])
        parser.add_argument('-style_weight', type=int)
        parser.add_argument('-content_weight', type=int)
        parser.add_argument('-tv_weight', type=int)

        args = parser.parse_args()

        for key, value in vars(args).items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid config parameter: {key}")
            if value is not None:
                setattr(self, key, value)

    def __str__(self):
        return str(vars(self))
