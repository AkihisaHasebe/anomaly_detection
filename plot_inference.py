from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

df = pd.read_csv('./logs/inference/inference.csv')
class_list = pd.read_csv('./dataset/vtuber/_database/vtuber.csv')

cm = confusion_matrix(df['y_true'].values, df['y_pred'].values)

fig, ax = plt.subplots(figsize=(14, 12))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'shrink': .3}, linewidths=.1, ax=ax)

ax.set(
    xticklabels=class_list['name'],
    yticklabels=class_list['name'],
    title='confusion matrix',
    ylabel='True label',
    xlabel='Predicted label'
)
params = dict(rotation=45, ha='center', rotation_mode='anchor')
plt.setp(ax.get_yticklabels(), **params)
plt.setp(ax.get_xticklabels(), **params)
plt.show()