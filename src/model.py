import torch
from torch import nn
from config import *

class Room_Classifier(nn.Module):

    def __init__(self, config):
        super(Room_Classifier, self).__init__()

        w = config.model_input_width
        h = config.model_input_height
        c = config.model_input_channels
        conv1, w, h, c = self.create_conv2d(w, h, c, 32, 3, 1, 0)
        conv2, w, h, c = self.create_conv2d(w, h, c, 64, 3, 1, 0)
        maxpool1, w, h, c = self.create_maxpool2d(w, h, c, 2)
        conv3, w, h, c = self.create_conv2d(w, h, c, 128, 3, 1, 0)
        maxpool2, w, h, c = self.create_maxpool2d(w, h, c, 2)

        self.convnet = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            maxpool1,
            conv3,
            nn.ReLU(),
            maxpool2
        )

        in_fea = w * h * c

        fc1 = nn.Linear(in_fea, in_fea)
        torch.nn.init.xavier_uniform_(fc1.weight)
        fc2 = nn.Linear(in_fea, config.no_of_classes)
        torch.nn.init.xavier_uniform_(fc2.weight)

        self.classifier = nn.Sequential(
            fc1,
            nn.Dropout(p=config.drop_prob),
            nn.ReLU(),
            fc2,
            nn.Dropout(p=config.drop_prob),
            nn.ReLU
        )

    def create_maxpool2d(self, w, h, c, kernel_size):
            # (W−F+2P) / S + 1
            offset = -kernel_size
            ow = w + offset + 1
            oh = h + offset + 1
            oc = c
            return nn.MaxPool2d(kernel_size, stride=0, padding=1), ow, oh, oc

    def create_conv2d(self, w, h, c, no_of_filters, kernel_size, stride, padding ):
        #(W−F+2P) / S + 1
        offset = -kernel_size + 2 * padding
        ow = int((w+offset) / stride) + 1
        oh = int((h+offset) / stride) + 1
        oc = no_of_filters
        return nn.Conv2d(c, no_of_filters, kernel_size, stride, padding), ow, oh, oc

    def forward(self, X):
        out = self.convnet(X)
        out = out.view(X.shape[0], -1)
        return self.classifier(out)
