import argparse


class Configs:
    def __init__(self, model='LeNet', attack='fsgm'):
        self.lr = 0.001
        self.num_epochs = 30
        self.device = 'cuda'
        self.seed = 1
        self.save_path = f'./{model}/model_save.pth'
        self.model = model
        self.attack = attack
        self.epsilon = 0.1

    def parse(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-num_epochs', type=int)
        parser.add_argument('-device', type=str)
        parser.add_argument('-lr', type=float)
        parser.add_argument('-seed', type=int)
        parser.add_argument('-save_path', type=str)
        parser.add_argument('-model', type=str, choices=['LeNet'])
        parser.add_argument('-attack', type=str, choices=['fsgm', 'fgm'])
        parser.add_argument('-epsilon', type=float)

        args = parser.parse_args()

        for key, value in vars(args).items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid config parameter: {key}")
            if value is not None:
                setattr(self, key, value)

    def __str__(self):
        return str(vars(self))
