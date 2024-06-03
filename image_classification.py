#
# Importing Library and Data
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
os.getcwd()

#
labels = pd.read_csv(r'dataset/aerial-cactus-identification/train.csv')
submission = pd.read_csv(r'dataset/aerial-cactus-identification/sample_submission.csv')
# print top most and bottom
print(labels.head())
print(labels.tail())
# value count
print(labels['has_cactus'].value_counts())
# visulize using pie chart
label = 'Has Cactus', 'Hasn\'t Cactus'
plt.figure(figsize = (8,8))
plt.pie(labels.groupby('has_cactus').size(), labels = label, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()


# Image Pre-processing
