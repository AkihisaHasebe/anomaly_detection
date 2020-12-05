import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from visdom import Visdom

import metrics
from arcface_resnet34 import Resnet34
from dataloader import CreateDataset, ImageTransform

cuda = torch.cuda.is_available()
if cuda:
    print('cuda available!')
    device = torch.device('cuda:0')

else:
    print('cuda non-available!')
    device = torch.device('cpu')

if __name__ == "__main__":
    random.seed(42)
    size = (224,224)
    img_num_class = 500
    train_ratio = 0.9
    classes = 20

    y_pred = np.array([])
    y_true = np.array([])

    char_path = list(Path('./dataset/vtuber/').glob('[!_]*'))
    weight_folder_model = Path('./logs/parameter/model/')
    weight_folder_metrics = Path('./logs/parameter/metrics/')

    num_train = int(img_num_class * train_ratio)

    target_char = 'amamiya'

    train_path = []
    valid_path = []

    for char in char_path:
        extract_path = random.sample(list(char.glob('*.png')), img_num_class)
        if target_char == char.name:
            train_path += extract_path[:num_train]

        valid_path += extract_path[num_train:]

    class_list = pd.read_csv('./dataset/vtuber/_database/vtuber.csv')
    office = {}

    for index, row in class_list.iterrows():
        office[row['name']] = row['office']


    transform = ImageTransform(size)
    train_dataset = CreateDataset(train_path,office,transform,'train')
    train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    model = Resnet34(classes=classes, return_GAP=True)
    model = model.to(device)
    model.load_state_dict(torch.load(weight_folder_model.joinpath('model_weight_epoch30.pth')))

    metric = metrics.ArcMarginProduct(512, classes, s=30, m=0.5, easy_margin=True)
    metric.to(device)
    metric.load_state_dict(torch.load(weight_folder_metrics.joinpath('model_weight_epoch30.pth')))

    images = []
    features = []
    model.eval()

    for (inputs, target) in tqdm(train_dataloader,desc='train'):
        inputs = inputs.to(device)
        # target = target.to(device).long()
        feature = model(inputs)
        feature = feature.to('cpu').detach().numpy().copy()[0]
        features.append(feature)

    features = np.hstack([np.array([str(p) for p in train_path]).reshape(-1,1), np.array(features)])
    df_features = pd.DataFrame(features)

    df_features.to_csv('./logs/inference/train_features.csv',index=False)