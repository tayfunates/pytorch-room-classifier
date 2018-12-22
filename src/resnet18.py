import torch
from torch import nn
import torchvision.models as models

class Resnet18_Room_Classifier(nn.Module):

    def __init__(self, config):
        super(Resnet18_Room_Classifier, self).__init__()
        self.convnet = models.resnet18(pretrained=True)
        self.convnet.fc = nn.Linear(512, config.no_of_classes)

    def forward(self, X):
        out = self.convnet(X)
        return out