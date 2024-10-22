import torch
from torch import nn
import torch.nn.functional as F

class MyOwnNeuralNetwork(nn.Module):
    def __init__(self):
        super(MyOwnNeuralNetwork, self).__init__()
        self.conv1 = nn.Sequential(
              nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
              nn.BatchNorm2d(64),
              #nn.ReLU(inplace=True)
          )

        self.conv2 = nn.Sequential(
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
              nn.BatchNorm2d(128),
              #nn.ReLU(inplace=True),
          )
        self.conv3 = nn.Sequential(
              nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
              nn.BatchNorm2d(128),
              #nn.ReLU(inplace=True),
          )

        self.fc1 = nn.Linear(1152, 25)
        #self.fc2 = nn.Linear(160, 25)
        self.fc3 = nn.Linear(25, 10)  ## Softmax layer ignored since the loss function defined is nn.CrossEntropy()

    def forward_before_softmax(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1152)
        x = self.fc1(x)
        return x
    
    
    def forward(self, x):
        x = self.forward_before_softmax(x)
        x = self.fc3(x)
        return x