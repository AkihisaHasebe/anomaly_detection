import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import models, transforms

import numpy as np
from tqdm import tqdm
from pathlib import Path
from visdom import Visdom
from PIL import Image

from torchsummary import  summary

class Resnet34(nn.Module):
    def __init__(self, classes, return_GAP=False):
        super(Resnet34, self).__init__()
        self.return_GAP = return_GAP

        resnet = models.resnet34(pretrained=True)

        self.feature = nn.Sequential(*list(resnet.children())[:-1])
        for param in list(self.feature.parameters())[:10]:
            param.requires_grad = False

        self.fc = nn.Linear(512,classes)

    def forward(self, x):
        x = self.feature(x)
        # x = self.fc(x.view(-1,512))
        return x.view(-1,512)

if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    if cuda:
        print('cuda available!')
        device = torch.device('cuda:0')

    else:
        print('cuda non-available!')
        device = torch.device('cpu')

    net = Resnet34(20)
    summary(net.to(device), (3,224,224))