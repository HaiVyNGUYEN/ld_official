import sys
import os
current = os.path.dirname(os.path.realpath('./'))
sys.path.append(current)

from utils.training_tools import train
from model_archi.model_archi_resnet18 import ResNet18_3

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset
import torch.nn.functional as F
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR


data_dir = 'dataset'
train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=False)
test_dataset  = torchvision.datasets.CIFAR10(data_dir, train=False, download=False)

train_transform = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                 (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
                                ])

test_transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                (0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784))])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

print(len(train_dataset),len(test_dataset))

batch_size= 128

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
## Defining optimizer and loss functions 


for sim in range(5):
    
    model = ResNet18_3().to(device)
    print(model)
    
    print(f"Starting to train model {sim+1}...")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    ## Defining optimizer and loss functions

    loss_fn = nn.CrossEntropyLoss()
    
    epochs = 200
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _train_batch_loss , _train_batch_accuracy = train(train_loader, model, loss_fn, optimizer, device)
        scheduler.step()
    
    torch.save(model.state_dict(), f'./cifar10_resnet18_200_epochs_{sim+1}')  ### respo of your choice

    print(datetime.now())

print("Done!")