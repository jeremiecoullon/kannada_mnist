from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.mymodel = nn.Sequential(
        nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.mymodel(x), dim=-1)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # "same"
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # H/2, W/2

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 7 * 7, 228),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(228, 10),
        )
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x


def build_model(model_type):
    if model_type=="mlp":
        return MLP()
    elif model_type=="cnn":
        return CNN()
    else:
        raise ValueError("model_type must be either 'mlp' or 'cnn'")
