import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset 
import numpy as np

class LightweightCNN(nn.Module):
    """
        A lightweight convolutional neural network (CNN) for image classification.
    """
    def __init__(self, dropout_rate=0.0):
        super(LightweightCNN, self).__init__()

        # 1st block: Conv -> ReLU -> MaxPool
        # Input: 3 channels (RGB), Output: 32 channels, Kernel size: 3x3, Padding: 1
        # Output: 16x16 feature map after pooling
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2nd block: Conv -> ReLU -> MaxPool
        # Input: 32 channels, Output: 64 channels, Kernel size: 3x3, Padding: 1
        # Output: 8x8 feature map after pooling
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 3rd block: Conv -> ReLU -> MaxPool
        # Input: 64 channels, Output: 128 channels, Kernel size: 3x3, Padding: 1
        # Output: 4x4 feature map after pooling
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer: Input: 128 * 4 * 4, Output: 256
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu4 = nn.ReLU()

        # Dropout value controlled by PSO and DBO algorithms
        self.dropout = nn.Dropout(dropout_rate)

        # Output layer: Input: 256, Output: 10 (for CIFAR-10 classes)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_data_loaders(batch_size=64, data_dir="./data", train_subset_ratio=1.0):
    """
    Download CIFAR-10, apply transformations, and return data loaders for training and testing. 

    Args:
        batch_size (int, optional): Hyperparameter searched in swarm algorithms. Defaults to 64.
        data_dir (str, optional): The directory where the CIFAR-10 data is stored. Defaults to "./data".
        train_subset_ratio (float, optional): The ratio of the training set to use. Useful for subsampling to speedup hyperparameter tuning. Defaults to 1.0.
    """

    # Traing set transformations
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Test set transformations
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)

    if train_subset_ratio < 1.0:
        # Subsample the training dataset
        num_train_samples = int(len(train_dataset) * train_subset_ratio)
        indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(42))[:num_train_samples]
        train_dataset = Subset(train_dataset, indices)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader