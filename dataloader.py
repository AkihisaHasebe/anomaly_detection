import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import random

from PIL import Image
from matplotlib import pyplot as plt

class ImageTransform():
    def __init__(self, size):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(size,scale=(0.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]),
            'valid': transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor()
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

class CreateDataset(data.Dataset):
    def __init__ (self, file_list, office, transform=None, phase='train', mode = 'personal'):
        self.file_list = file_list
        self.office = office
        self.personal_key = list(office.keys())
        self.transform = transform
        self.phase = phase
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)
        img_ = self.transform(img, phase=self.phase)

        if self.mode == 'office':
            target = self.office[img_path.parents[0].name]
            if target == 'nijisanji':
                target = 0.0
            elif target == 'hololive':
                target = 1.0

        elif self.mode == 'personal':
            target = self.personal_key.index(img_path.parents[0].name)


        return img_, target

if __name__ == "__main__":
    random.seed(42)
    size = (128,128)

    valid_name = ['subaru','suisei','suzuhara','yorumi']
    char_path = Path('./dataset/vtuber/').glob('[!_]*')

    train_path = []
    valid_path = []
    for char in char_path:
        if char.name in valid_name:
            valid_path += random.sample(list(char.glob('*.png')),500)
        else:
            train_path += random.sample(list(char.glob('*.png')),500)

    class_list = pd.read_csv('./dataset/vtuber/_database/vtuber.csv')
    office = {}

    for index, row in class_list.iterrows():
        office[row['name']] = row['office']

    transform = ImageTransform(size)
    
    train_dataset = CreateDataset(train_path,office,transform,'train','personal')

