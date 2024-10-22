import sys
import os
current = os.path.dirname(os.path.realpath('./'))
sys.path.append(current)
from utils.training_tools import train, validation, copy_state_dict
from model_archi.fashion_mnist_archi import MyOwnNeuralNetwork

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.optim.lr_scheduler import StepLR


data_dir = './dataset'
train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=False)
test_dataset  = torchvision.datasets.FashionMNIST(data_dir, train=False, download=False)
train_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.286,), (0.3205,))])

test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.286,), (0.3205,))])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)

train_data, val_data = random_split(train_dataset, [50000, 10000])
batch_size = 128

train_loader = DataLoader(train_data, batch_size=batch_size)
valid_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

    
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")

for i in range(5):
    
    model = MyOwnNeuralNetwork().to(device)
    print(model)
    
    print(f"Starting to train model {i+1}...")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.2)

    ## Defining optimizer and loss functions

    loss_fn = nn.CrossEntropyLoss()

    train_batch_loss = []
    train_batch_accuracy = []
    valid_batch_accuracy = []
    valid_batch_loss = []
    train_epoch_no = []
    valid_epoch_no = []
    correct = 0
    epochs = 30
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _train_batch_loss , _train_batch_accuracy = train(train_loader, model, loss_fn, optimizer, device)
        _valid_batch_loss , _valid_batch_accuracy, correct_temp = validation(valid_loader, model, loss_fn, device)
        if correct_temp >= correct:
            state = copy_state_dict(model)
            correct = correct_temp

        for i in range(len(_train_batch_loss)):
            train_batch_loss.append(_train_batch_loss[i])
            train_batch_accuracy.append(_train_batch_accuracy[i])
            train_epoch_no.append( t + float((i+1)/len(_train_batch_loss)))

        for i in range(len(_valid_batch_loss)):
            valid_batch_loss.append(_valid_batch_loss[i])
            valid_batch_accuracy.append(_valid_batch_accuracy[i])
            valid_epoch_no.append( t + float((i+1)/len(_valid_batch_loss)))
        scheduler.step()

    torch.save(state, f'./fashion_mnist_net_30_epochs_{i+1}')

    
    
print("Done!")

