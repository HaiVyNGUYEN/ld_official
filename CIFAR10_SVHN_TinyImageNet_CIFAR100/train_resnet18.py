import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset
import torch.nn.functional as F
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from archi.model_archi_resnet18 import ResNet18_3


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


### Defining training and evaluation routine

def train(dataloader, model, loss_fn, optimizer):
    
    # Total size of dataset for reference
    size = 0
    
    # places your model into training mode
    model.train()
    
    # loss batch
    batch_loss = {}
    batch_accuracy = {}
    
    correct = 0
    _correct = 0
    
    
    
    # Gives X , y for each batch
    for batch, (X, y) in enumerate(dataloader):
        
        # Converting device to cuda
        X, y = X.to(device), y.to(device)
        model.to(device)
        
        # Compute prediction error / loss
        # 1. Compute y_pred 
        # 2. Compute loss between y and y_pred using selectd loss function
        
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        # Backpropagation on optimizing for loss
        # 1. Sets gradients as 0 
        # 2. Compute the gradients using back_prop
        # 3. update the parameters using the gradients from step 2
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        _correct = (y_pred.argmax(1) == y).float().sum().item()
        _batch_size = len(X)
        
        correct += _correct
        
        # Updating loss_batch and batch_accuracy
        batch_loss[batch] = loss.item()
        batch_accuracy[batch] = _correct/_batch_size
        
        size += _batch_size
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}]")
    
    correct/=size
    print(f"Train Accuracy: {(100*correct):>0.1f}%")
    
    return batch_loss , batch_accuracy

def validation(dataloader, model, loss_fn):
    
    # Total size of dataset for reference
    size = 0
    num_batches = len(dataloader)
    
    # Setting the model under evaluation mode.
    model.eval()

    test_loss, correct = 0, 0
    
    _correct = 0
    _batch_size = 0
    
    batch_loss = {}
    batch_accuracy = {}
    
    with torch.no_grad():
        
        # Gives X , y for each batch
        for batch , (X, y) in enumerate(dataloader):
            
            X, y = X.to(device), y.to(device)
            model.to(device)
            pred = model(X)
            batch_loss[batch] = loss_fn(pred, y).item()
            test_loss += batch_loss[batch]
            _batch_size = len(X)
            
            _correct = (pred.argmax(1) == y).float().sum().item()
            correct += _correct
            
            size+=_batch_size
            batch_accuracy[batch] = _correct/_batch_size
            
            
            
    
    ## Calculating loss based on loss function defined
    test_loss /= num_batches
    
    ## Calculating Accuracy based on how many y match with y_pred
    correct /= size
    
    print(f"Valid Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return batch_loss , batch_accuracy, correct

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
        _train_batch_loss , _train_batch_accuracy = train(train_loader, model, loss_fn, optimizer)
        scheduler.step()
    
    torch.save(model.state_dict(), f'./cifar10_resnet18_200_epochs_{sim+1}')

    print(datetime.now())

print("Done!")