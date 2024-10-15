import numpy as np # this module is useful to work with numerical arrays
import pandas as pd
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
data_dir = './dataset'

# download fashionMNIST

train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True)

# download only the testset of MNIST

test_dataset_mnist  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

# snipcode for computing mean ad std

# train_dataset.transform = torchvision.transforms.ToTensor()
# loader = torch.utils.data.DataLoader(train_dataset , batch_size=128, num_workers=2, shuffle=True)
# mean = 0.
# std = 0.
# nb_samples = 0
# for (data, y) in loader:
#     batch_samples = data.size(0)
#     data = data.view(batch_samples, data.size(1), -1)#.float()
#     mean += data.mean(2).sum(0)
#     std += data.std(2).sum(0)
#     nb_samples += batch_samples

# mean /= nb_samples
# std /= nb_samples
# print(mean,std)