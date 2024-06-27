# Cactus Image Classification using Convolutional Neural Network (CNN)

This repository contains the implementation of a Convolutional Neural Network (CNN) model for image classification. The primary task of this project is to identify whether an image contains a cactus or not.

## Table of Contents

1. [Importing Libraries](#importing-libraries)
2. [Loading and Visualizing the Data](#loading-and-visualizing-the-data)
3. [Preprocessing the Data](#preprocessing-the-data)
4. [Setting up the Model](#setting-up-the-model)
5. [Training the Model](#training-the-model)
6. [Testing the Model](#testing-the-model)
7. [Saving the Model](#saving-the-model)
8. [Plotting the Loss](#plotting-the-loss)

## Importing Libraries

The project begins by importing necessary libraries such as pandas, matplotlib, torch, torchvision, sklearn, os, and numpy.

## Loading and Visualizing the Data

The code loads CSV files containing labels and sample submissions. It defines paths to the training and test images. It also displays the first and last few rows of the labels dataframe, counts the number of images with and without cactus, and plots a pie chart to visualize the distribution of images.

## Preprocessing the Data

The code defines a function to display images with optional normalization. It also creates a custom Dataset class for loading images and their labels. It defines transformations for training, validation, and test datasets. The dataset is split into training and validation sets.

## Setting up the Model

The code sets up the CNN model with two convolutional layers, a dropout layer, and two fully connected layers. It defines the loss function and the optimizer. It also sets up DataLoader objects for training, validation, and testing.

## Training the Model

The code trains the model using a training loop. It computes the loss for each epoch and updates the weights of the model. It also validates the model and computes the validation loss.

## Testing the Model

The code tests the model on the validation set and computes the accuracy of the model.

## Saving the Model

The code saves the trained model for future use.

## Plotting the Loss

Finally, the code plots the training and validation loss for each epoch.

