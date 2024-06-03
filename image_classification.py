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


labels = pd.read_csv(r'dataset/aerial-cactus-identification/train.csv')
submission = pd.read_csv(r'dataset/aerial-cactus-identification/sample_submission.csv')

