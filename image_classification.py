# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import matplotlib.image as img
import numpy as np

# Set to display plots inline
# %matplotlib inline

# Check the current working directory
print(os.getcwd())

# Load the CSV files containing labels and sample submissions
labels = pd.read_csv(r'dataset/aerial-cactus-identification/train.csv')
submission = pd.read_csv(r'dataset/aerial-cactus-identification/sample_submission.csv')

# Define paths to the training and test images
train_path = r'dataset/aerial-cactus-identification/train/'
test_path = r'dataset/aerial-cactus-identification/test/'

# Display the first few rows of the labels dataframe
print('the first few rows of the labels dataframe',labels.head())

# Display the last few rows of the labels dataframe
print('the last few rows of the labels dataframe',labels.tail())

# Count the number of images with and without cactus
print('the number of images with and without cactus',labels['has_cactus'].value_counts())

# Plot a pie chart to visualize the distribution of images with and without cactus
label = 'Has Cactus', 'Hasn\'t Cactus'
plt.figure(figsize=(8, 8))
plt.pie(labels.groupby('has_cactus').size(), labels=label, autopct='%1.1f%%', shadow=True, startangle=90)
plt.show()

# Display sample images of cacti
fig, ax = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(labels[labels['has_cactus'] == 1]['id'][-5:]):
    path = os.path.join(train_path, idx)
    ax[i].imshow(img.imread(path))

# Display sample images without cacti
fig, ax = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(labels[labels['has_cactus'] == 0]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[i].imshow(img.imread(path))


# Function to display images with optional normalization
def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax


# Custom Dataset class for loading images and their labels
class CactiDataset(Dataset):
    def __init__(self, data, path, transform=None):
        super().__init__()
        self.data = data.values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, label = self.data[index]
        img_path = os.path.join(self.path, img_name)
        image = img.imread(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# Define transformations for training, validation, and test datasets
means = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(means, std)])

test_transform = transforms.Compose([transforms.ToPILImage(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(means, std)])

valid_transform = transforms.Compose([transforms.ToPILImage(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(means, std)])

# Split the dataset into training and validation sets
train, valid_data = train_test_split(labels, stratify=labels.has_cactus, test_size=0.2)

# Create dataset objects for training, validation, and testing
train_data = CactiDataset(train, train_path, train_transform)
valid_data = CactiDataset(valid_data, train_path, valid_transform)
test_data = CactiDataset(submission, test_path, test_transform)

# Hyperparameters for training
num_epochs = 35
num_classes = 2
batch_size = 25
learning_rate = 0.001

# Check if GPU is available and set the device accordingly
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Create DataLoader objects for training, validation, and testing
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0)

# Display sample training images
trainimages, trainlabels = next(iter(train_loader))
fig, axes = plt.subplots(figsize=(12, 12), ncols=5)
print('training images')
for i in range(5):
    axe1 = axes[i]
    imshow(trainimages[i], ax=axe1, normalize=False)
print(trainimages[0].size())


# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


# Instantiate the CNN model and print its architecture
model = CNN()
print(' the CNN model architecture :', model)

# Define the loss function and the optimizer
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
# %%time
train_losses = []
valid_losses = []

for epoch in range(1, num_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0

    # Training phase
    model.train()
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()  # Clear the gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        train_loss += loss.item() * data.size(0)  # Update training loss

    # Validation phase
    model.eval()
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Compute loss
        valid_loss += loss.item() * data.size(0)  # Update validation loss

    # Calculate average losses
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # Print training/validation statistics
    print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tValidation Loss: {valid_loss:.6f}')

# Test the model
model.eval()  # Disable dropout for evaluation
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Test Accuracy of the model: {100 * correct / total} %')

# Save the trained model
torch.save(model.state_dict(), 'model.ckpt')

# Plot the training and validation loss
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)
plt.show()
