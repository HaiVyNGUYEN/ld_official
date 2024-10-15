import numpy as np # this module is useful to work with numerical arrays
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from model_archi import MyOwnNeuralNetwork

data_dir = './dataset'
train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=False)
test_dataset  = torchvision.datasets.FashionMNIST(data_dir, train=False, download=False)
train_transform = transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.286,), (0.3205,))])

test_transform = transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.286,), (0.3205,))])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)

train_data, val_data = random_split(train_dataset, [50000, 10000])
batch_size = 128

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)


#### Defining model


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
    
    
#### Defining training scheme


def train(dataloader, model, loss_fn, optimizer, device = "cuda"):

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

        _correct = (y_pred.argmax(1) == y).type(torch.float).sum().item()
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

def validation(dataloader, model, loss_fn, device = "cuda"):

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

            _correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += _correct

            size+=_batch_size
            batch_accuracy[batch] = _correct/_batch_size




    ## Calculating loss based on loss function defined
    test_loss /= num_batches

    ## Calculating Accuracy based on how many y match with y_pred
    correct /= size

    print(f"Valid Error: \n Accuracy: {(100*correct):>0.3f}%, Avg loss: {test_loss:>8f} \n")

    return batch_loss , batch_accuracy, correct

def accuracy_evaluation(dataloader, model, device = "cuda"):

    # Total size of dataset for reference
    size = 0
    num_batches = len(dataloader)

    # Setting the model under evaluation mode.
    model.eval()

    correct =  0

    _correct = 0
    _batch_size = 0
    batch_accuracy = {}

    with torch.no_grad():

        # Gives X , y for each batch
        for batch , (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)
            
            model.to(device)
            pred = model(X)
            
            _batch_size = len(X)

            _correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += _correct
            size+=_batch_size
            batch_accuracy[batch] = _correct/_batch_size


    ## Calculating Accuracy based on how many y match with y_pred
    correct /= size

    return correct

def copy_state_dict(model):
    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()
    return old_state_dict

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")


from torch.optim.lr_scheduler import StepLR

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

