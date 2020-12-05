import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from visdom import Visdom
from PIL import Image
import random

from arcface_resnet34 import Resnet34
import metrics

from dataloader import ImageTransform, CreateDataset

cuda = torch.cuda.is_available()
if cuda:
    print('cuda available!')
    device = torch.device('cuda:0')

else:
    print('cuda non-available!')
    device = torch.device('cpu')

def init_seed(rand_seed, reproduction = False):
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)
    random.seed(rand_seed)

    if reproduction == True:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train(model, metric, optimizer, criterion, dataloaders):
    train_dataloader, valid_dataloader = dataloaders

    train_loss = 0.0
    model.train()
    for (inputs, target) in tqdm(train_dataloader, leave=False, desc='[train]'):
        inputs, target = inputs.to(device), target.to(device).long()
        optimizer.zero_grad()
        features = model(inputs)
        outputs = metric(features, target)
        loss = criterion(outputs, target)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    valid_loss = 0.0
    model.eval()
    for (inputs, target) in tqdm(valid_dataloader,leave=False, desc='[valid]'):
        inputs, target = inputs.to(device), target.to(device).long()
        optimizer.zero_grad()
        features = model(inputs)
        outputs = metric(features, target)
        loss = criterion(outputs, target)

        valid_loss += loss.item()

    train_loss_ = train_loss*train_dataloader.batch_size/len(train_dataloader.dataset)
    valid_loss_ = valid_loss*valid_dataloader.batch_size/len(valid_dataloader.dataset)

    return model, train_loss_, valid_loss_


if __name__ == "__main__":
    random.seed(42)
    init_seed(42)
    size = (224,224)
    batch_size = 32
    img_num_class = 500
    train_ratio = 0.9
    epochs = 30
    classes = 0

    valid_name = ['subaru','suisei','suzuhara','yorumi']
    char_path = Path('./dataset/vtuber/').glob('[!_]*')
    weight_folder_model = Path('./logs/parameter/model')
    if not weight_folder_model.exists():
        weight_folder_model.mkdir(parents=True)

    weight_folder_metrics = Path('./logs/parameter/metrics')
    if not weight_folder_metrics.exists():
        weight_folder_metrics.mkdir(parents=True)

    num_train = int(img_num_class * train_ratio)
    train_path = []
    valid_path = []
    # for char in char_path:
    #     if char.name in valid_name:
    #         valid_path += random.sample(list(char.glob('*.png')),500)
    #     else:
    #         train_path += random.sample(list(char.glob('*.png')),500)

    # classes = 1

    for char in char_path:
        extract_path = random.sample(list(char.glob('*.png')), img_num_class)
        train_path += extract_path[:num_train]
        valid_path += extract_path[num_train:]
        classes += 1

    class_list = pd.read_csv('./dataset/vtuber/_database/vtuber.csv')
    office = {}

    for index, row in class_list.iterrows():
        office[row['name']] = row['office']


    transform = ImageTransform(size)
    train_dataset = CreateDataset(train_path,office,transform,'train',mode='personal')
    valid_dataset = CreateDataset(valid_path,office,transform,'valid',mode='personal')

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = data.DataLoader(valid_dataset, batch_size=4, shuffle=False)

    log_dir = './logs'
    vis = Visdom()

    model = Resnet34(classes = classes)
    model = model.to(device)

    metric = metrics.ArcMarginProduct(512, classes, s=30, m=0.5, easy_margin=True)
    metric.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params':model.parameters()}, {'params':metric.parameters()}],
                            lr=1e-3,weight_decay=5e-4)

    for epoch in tqdm(range(epochs)):
        model, train_loss, valid_loss = train(
            model, metric, optimizer, criterion, (train_dataloader, valid_dataloader)
        )
        # train_loss_list.append(train_loss)
        # valid_loss_list.append(valid_loss)

        vis.line(X=np.array([epoch]), Y=np.array([train_loss]), win='loss', name='train_loss', update='append')
        vis.line(X=np.array([epoch]), Y=np.array([valid_loss]), win='loss', name='valid_loss', update='append')

        torch.save(model.state_dict(),weight_folder_model.joinpath(f'model_weight_epoch{epoch+1}.pth'))
        torch.save(metric.state_dict(),weight_folder_metrics.joinpath(f'model_weight_epoch{epoch+1}.pth'))