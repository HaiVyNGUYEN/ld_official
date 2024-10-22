import sys
import os
current = os.path.dirname(os.path.realpath('./'))
sys.path.append(current)
from utils.training_tools import train, validation

from model_archi.model_archi_wideresnet_SN import wrn

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset
import torch.nn.functional as F
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
import torch
from torch import nn

### Download data

# data_dir = 'dataset'
# train_dataset = torchvision.datasets.CIFAR100(data_dir, train=True, download=True)
# test_dataset  = torchvision.datasets.CIFAR100(data_dir, train=False, download=True)

### Defining datasetss

data_dir = 'dataset'
train_dataset = torchvision.datasets.CIFAR100(data_dir, train=True, download=False)
test_dataset  = torchvision.datasets.CIFAR100(data_dir, train=False, download=False)

train_transform = transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                 (0.49139968, 0.48215841, 0.44653091), (0.2023, 0.1994, 0.2010)),
                                ])

test_transform = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(
                                (0.49139968, 0.48215841, 0.44653091), (0.2023, 0.1994, 0.2010))])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

print(len(train_dataset),len(test_dataset))

batch_size= 128

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)


from datetime import datetime

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
## Defining optimizer and loss functions 
now = datetime.now()
for i in range(1,6):
    print(('Try ',i))

    model = wrn(spectral_normalization=True,num_classes=100).to(device)
    
    
    #model = wrn(spectral_normalization=False,num_classes=100).to(device)
    #print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                    momentum=0.9, nesterov=False, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    

    loss_fn = nn.CrossEntropyLoss()

    epochs = 350

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _train_batch_loss , _train_batch_accuracy = train(train_loader, model, loss_fn, optimizer, device)
        _valid_batch_loss , _valid_batch_accuracy, correct_temp = validation(test_loader, model, loss_fn, device)

        #if (t+1)%5==0:
        print(datetime.now()-now)

        scheduler.step()

    torch.save(model.state_dict(), f'./cifar100_SN_wideresnet_350_epochs_{i}') ### set your respo of choice

    print(datetime.now())

print("Done!")