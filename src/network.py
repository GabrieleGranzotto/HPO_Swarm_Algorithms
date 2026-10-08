import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset 
import numpy as np
import yaml

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


class NN_Environment:
    """
    Environment for evaluating the performance of a CNN model on the CIFAR-10 dataset.
    """
    def __init__(self, model=LightweightCNN, device='cpu'):
        with open("config.yml", 'r') as file:
            config = yaml.safe_load(file)
        self.dataset_path = config["dataset"]["data_dir"]  # Path to the dataset
        self.train_subset_ratio = config["dataset"]["train_subset_ratio"]  # Ratio of training set to use
        self.num_epochs = config["training_pipeline"]["proxy_epochs"]  # Number of epochs for training
        
        self.lr_log_bounds = config["search_space"]["lr_log_bounds"]  # Learning rate bounds
        self.dropout_bounds = config["search_space"]["dropout_bounds"]  # Dropout rate bounds
        self.batch_size_choices = config["search_space"]["batch_size_choices"]  # Batch size bounds

        self.model = model
        self.train_dataset, self.test_dataset = self._download_data()
        self.problem_type = "maximization"
        self.device = device

    def evaluate(self, parameters_normalized):
        """
        Evaluate the CNN model with the given hyperparameters.

        Args:
            parameters (list): A list containing [learning rate, dropout_rate, batch_size].

        Returns:
            float: The accuracy of the model on the test set.
        """

        [learning_rate_normalized, dropout_rate_normalized, batch_size_normalized] = parameters_normalized

        # Denormalize the hyperparameters
        learning_rate = self._denormalize(learning_rate_normalized, "exp", self.lr_log_bounds[0], self.lr_log_bounds[-1])
        dropout_rate = self._denormalize(dropout_rate_normalized, "float", self.dropout_bounds[0], self.dropout_bounds[-1])
        batch_size = self._denormalize(batch_size_normalized, "int", self.batch_size_choices[0], self.batch_size_choices[-1])

        model = self.model(dropout_rate=dropout_rate).to(self.device)
        train_loader, test_loader = self._get_data_loaders(self.train_dataset, self.test_dataset, batch_size=batch_size)
        self._train_model(model, train_loader, learning_rate, self.num_epochs)
        return self._test_model(model, test_loader)

    def _denormalize(self, value, param_type, min_value, max_value):
        """
        Denormalize a value from [0, 1] to its original range.

        Args:
            value (float): The normalized value in the range [0, 1].
            param_type (str): The type of the parameter ("float", "exp", or "int").
            min_value (float): The minimum value of the original range.
            max_value (float): The maximum value of the original range.

        Returns:
            float: The denormalized value in the original range.
        """
        if param_type == "float":
            return min_value + (max_value - min_value) * value
        elif param_type == "exp":
            return 10**(min_value + (max_value - min_value) * value)
        elif param_type == "int":
            denormalized_value = round(min_value + (max_value - min_value) * value)
            # compute logarithm of base 2
            final_value = 2**round(np.log2(denormalized_value))
            return int(final_value)
        else:
            raise ValueError("param_type must be 'float', 'exp', or 'int'")

    def _train_model(self, model, train_loader, learning_rate, num_epochs=5):
        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def _test_model(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def _download_data(self):
        """
        Download CIFAR-10, apply transformations. 

        Args:
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
        train_dataset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root=self.dataset_path, train=False, download=True, transform=transform_test)

        if self.train_subset_ratio < 1.0:
            # Subsample the training dataset
            num_train_samples = int(len(train_dataset) * self.train_subset_ratio)
            indices = torch.randperm(len(train_dataset), generator=torch.Generator().manual_seed(42))[:num_train_samples]
            train_dataset = Subset(train_dataset, indices)
            
        return train_dataset, test_dataset

    def _get_data_loaders(self, train_dataset, test_dataset, batch_size=64):
        """
        Create data loaders for training and testing datasets.

        Args:
            train_dataset (Dataset): The training dataset.
            test_dataset (Dataset): The testing dataset.
            batch_size (int, optional): The batch size for the data loaders. Defaults to 64.
        """
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        return train_loader, test_loader
