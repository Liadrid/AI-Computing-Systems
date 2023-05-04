import argparse


class Configs:
    def __init__(self):
        self.lr = 0.001
        self.num_epochs = 10
        self.device = 'cpu'
        self.seed = 1
        self.save_path = './model_save'
        self.model = 'LeNet'
        self.attack = 'fsgm'
        self.epsilon = 0.05

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
