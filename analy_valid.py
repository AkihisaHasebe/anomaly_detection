import random
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
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
    classes = 0

    y_pred = np.array([])
    y_true = np.array([])

    char_path = list(Path('./dataset/vtuber/').glob('[!_]*'))
    weight_folder_model = Path('./logs/parameter/model/')
    weight_folder_metrics = Path('./logs/parameter/metrics/')

    num_train = int(img_num_class * train_ratio)
    valid_path = []

    for char in char_path:
        extract_path = random.sample(list(char.glob('*.png')), img_num_class)
        valid_path += extract_path[num_train:]
        # valid_path.append(extract_path[-1])
        classes = 20

    # valid_path = random.sample(valid_path, 20)
    class_list = pd.read_csv('./dataset/vtuber/_database/vtuber.csv')
    office = {}

    for index, row in class_list.iterrows():
        office[row['name']] = row['office']


    transform = ImageTransform(size)
    valid_dataset = CreateDataset(valid_path,office,transform,'valid')

    valid_dataloader = data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

    model = Resnet34(classes=classes, return_GAP=True)
    model = model.to(device)
    model.load_state_dict(torch.load(weight_folder_model.joinpath('model_weight_epoch30.pth')))

    metric = metrics.ArcMarginProduct(512, classes, s=30, m=0.5, easy_margin=True)
    metric.to(device)
    metric.load_state_dict(torch.load(weight_folder_metrics.joinpath('model_weight_epoch30.pth')))

    images = []
    features = []
    model.eval()
    for (inputs, target) in tqdm(valid_dataloader,desc='valid'):
        inputs = inputs.to(device)
        target = target.to(device).long()
        feature = model(inputs)
        feature = feature.to('cpu').detach().numpy().copy()[0]
        features.append(feature)

    features = np.hstack([np.array([str(p) for p in valid_path]).reshape(-1,1), np.array(features)])
    df_features = pd.DataFrame(features)

    df_features.to_csv('./logs/inference/features.csv',index=False)
    #     outputs = metric(feature,target)

    #     outputs = outputs.to('cpu').detach().numpy().copy()

    #     y_true = np.append(y_true,target.to('cpu').detach().numpy().copy().ravel())
    #     y_pred = np.append(y_pred,np.argmax(outputs,axis=1).ravel())

    # df = pd.DataFrame({'image path':valid_path,'y_true':y_true,'y_pred':y_pred})

    # df.to_csv('./logs/inference/inference.csv',index=None)

    # cm = confusion_matrix(y_true, y_pred)

    # fig, ax = plt.subplots(figsize=(7, 6))

    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': .3}, linewidths=.1, ax=ax)

    # ax.set(
    #     xticklabels=list(label_to_class.keys()),
    #     yticklabels=list(label_to_class.keys()),
    #     title='confusion matrix',
    #     ylabel='True label',
    #     xlabel='Predicted label'
    # )
    # params = dict(rotation=45, ha='center', rotation_mode='anchor')
    # plt.setp(ax.get_yticklabels(), **params)
    # plt.setp(ax.get_xticklabels(), **params)
    # plt.show()
