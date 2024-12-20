import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, pixels, input_channels=3, num_classes=6):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layer: input_channels -> 16 feature maps
        self.feature_maps = 16
        self.conv1 = nn.Conv2d(input_channels, self.feature_maps, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # Assuming input images are 28x28 (e.g., MNIST)
        # After conv1 + pool, the feature map size is 16 x 14 x 14
        self.fc1 = nn.Linear(self.feature_maps * int(pixels[0]/2) * int(pixels[1]/2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Convolutional layer + activation
        x = F.relu(self.conv1(x))
        # Max pooling
        x = self.pool(x)
        # Flatten the feature maps
        x = x.view(x.size(0), -1)
        # Fully connected layers + activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x