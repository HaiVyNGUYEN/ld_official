import sys
import os
current = os.path.dirname(os.path.realpath('./'))
sys.path.append(current)
from utils.ld_tools import *
import random
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Subset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math
from sklearn.cluster import KMeans


### Downloading dataset

# data_dir = './dataset'
# train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True)
# test_dataset  = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True)


### Calculate mean and std

# train_dataset.transform = torchvision.transforms.ToTensor()
# loader = torch.utils.data.DataLoader(train_dataset , batch_size=128, num_workers=2, shuffle=True)
# mean = 0.
# std = 0.
# nb_samples = 0.
# for (data, y) in loader:
#     batch_samples = data.size(0)
#     data = data.view(batch_samples, data.size(1), -1)#.float()
#     mean += data.mean(2).sum(0)
#     std += data.std(2).sum(0)
#     nb_samples += batch_samples

# mean /= nb_samples
# std /= nb_samples
# print(mean,std)


data_dir = './dataset'
train_dataset0 = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset0  = torchvision.datasets.MNIST(data_dir, train=False, download=False)
train_transform = transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])

test_transform = transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))])

train_dataset0.transform = train_transform
test_dataset0.transform = test_transform

train_dataset = Subset(train_dataset0, np.where(np.array([u[1] for u in train_dataset0])!=9)[0])
test_dataset = Subset(test_dataset0, np.where(np.array([u[1] for u in test_dataset0])!=9)[0])
hold_out_test_dataset = Subset(test_dataset0, np.where(np.array([u[1] for u in test_dataset0])==9)[0])

print(len(train_dataset),len(test_dataset), len(hold_out_test_dataset))

batch_size=128

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
test_loader_out = torch.utils.data.DataLoader(hold_out_test_dataset, batch_size=batch_size,shuffle=False)

### Defining model


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
        self.fc3 = nn.Linear(25, 9)  ## Softmax layer ignored since the loss function defined is nn.CrossEntropy()

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
    
### Defining valid/test loop
 
    
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


def accuracy_evaluation(dataloader, model):

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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
## Defining optimizer and loss functions 
for i in range(1,6):
    print(('Try ',i))

    model = MyOwnNeuralNetwork().to(device)
    #print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.2)

    loss_fn = nn.CrossEntropyLoss()

    epochs = 30
    from datetime import datetime
    now = datetime.now()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _train_batch_loss , _train_batch_accuracy = train(train_loader, model, loss_fn, optimizer)
        _valid_batch_loss , _valid_batch_accuracy, correct_temp = validation(test_loader, model, loss_fn)

        if (t+1)%5==0:
            print(datetime.now()-now)

    torch.save(model.state_dict(), f'./mnist_hold_one_out_30_epochs_{i}')
    #torch.save(model.state_dict(), f'./mnist_hold_one_out_20_epochs_{i}')

    print(datetime.now())

print("Done!")


### LD method

now = datetime.now()
auc_LD = []
for sim in range(1,6):
    
    print('Try:', sim)
    # load model
    
    model = MyOwnNeuralNetwork().to(device)
    model.load_state_dict(torch.load(f'./mnist_hold_one_out_30_epochs_{sim}'))
    model.eval()

    # compute feature to test
    feature_test_in = None
    with torch.no_grad():
        for loader in test_loader:
            X,_ = loader
            X = X.cuda()
            if feature_test_in is None:
                feature_test_in = np.array(model.forward_before_softmax(X).cpu())
            else:
                feature_test_in = np.concatenate((feature_test_in,np.array(model.forward_before_softmax(X).cpu())))
    print(feature_test_in.shape)

    feature_test_out = None
    with torch.no_grad():
        for loader in test_loader_out:
            X,_ = loader
            X = X.cuda()
            if feature_test_out is None:
                feature_test_out = np.array(model.forward_before_softmax(X).cpu())
            else:
                feature_test_out = np.concatenate((feature_test_out,np.array(model.forward_before_softmax(X).cpu())))
    print(feature_test_out.shape)

    # compuute feature of training set
    # feature = np.zeros((9,6000,25))
    feature = []
    for i in range(9):
        print('Group: ',i)
        image_group_prime = [u for u in train_dataset if u[1]==i]
        image_group = [u[0].cuda() for u in image_group_prime if \
                       model(u[0].cuda().unsqueeze(0)).argmax(1).item()==i]
        image_group = torch.stack(image_group)
        print(image_group.shape)
        with torch.no_grad():
            feature += [np.array(model.forward_before_softmax(image_group).detach().cpu()).reshape(-1,25)]

    #print(feature.shape)


    # K-mean
    print("K-mean")
    num_centers = 500
    centers = []
    for i in range(9):
        km = KMeans(n_clusters=num_centers,n_init=10)
        y_km = km.fit_predict(feature[i])
        c = np.array(km.cluster_centers_)
        print(c.shape)
        centers += [c]
    centers = np.array(centers)
    print(centers.shape)
    
    
    
    alpha=7

    # Internal fermat distance

    dist_mat_fermat_reduced = np.zeros((9,num_centers,num_centers))
    for i in range(9):
        print('Group:',i)
        dist_mat_group = distance_matrix(centers[i],centers[i])
        dist_mat_fermat_reduced[i] = fermat_function(dist_mat_group,alpha=alpha)


    # LD

    lens_depth_in_reduced = np.zeros((len(feature_test_in),9))

    # in

    batch_size = 250
    num_batches_in = int(len(feature_test_in)/batch_size)
    for i in range(9):
        dist_mat =  distance_matrix(centers[i],feature_test_in)
        print('Group',i)
        for k in range(num_batches_in):
            print('Processing batch:',k)
            dist_ext = fermat_function_ext_simultaneous(dist_mat_fermat_reduced[i],\
                                                        dist_mat[:,batch_size*k:batch_size*(k+1)].T,alpha=alpha)
            lens_depth_in_reduced[batch_size*k:batch_size*(k+1),i] = \
                                lens_depth_ext_function_simultaneous(dist_mat_fermat_reduced[i], dist_ext)
        if num_batches_in*batch_size < len(feature_test_in):
            print('Processing batch:',num_batches_in)
            dist_ext = fermat_function_ext_simultaneous(dist_mat_fermat_reduced[i],\
                                                        dist_mat[:,num_batches_in*batch_size:].T,alpha=alpha)
            lens_depth_in_reduced[num_batches_in*batch_size:,i] = \
                                lens_depth_ext_function_simultaneous(dist_mat_fermat_reduced[i], dist_ext)

    # out

    lens_depth_out_reduced = np.zeros((len(feature_test_out),9))
    num_batches_out = int(len(feature_test_out)/batch_size)

    for i in range(9):
        dist_mat =  distance_matrix(centers[i],feature_test_out)
        print('Group',i)
        for k in range(num_batches_out):
            print('Processing batch:',k)
            dist_ext = fermat_function_ext_simultaneous(dist_mat_fermat_reduced[i],\
                                                        dist_mat[:,batch_size*k:batch_size*(k+1)].T,alpha=alpha)
            lens_depth_out_reduced[batch_size*k:batch_size*(k+1),i] = \
                                lens_depth_ext_function_simultaneous(dist_mat_fermat_reduced[i], dist_ext)
        if num_batches_out*batch_size < len(feature_test_out):
            print('Processing batch:',num_batches_out)
            dist_ext = fermat_function_ext_simultaneous(dist_mat_fermat_reduced[i],\
                                                        dist_mat[:,num_batches_out*batch_size:].T,alpha=alpha)
            lens_depth_out_reduced[num_batches_out*batch_size:,i] = \
                                lens_depth_ext_function_simultaneous(dist_mat_fermat_reduced[i], dist_ext)
    LD_in_reduced = np.max(lens_depth_in_reduced,axis=1)
    LD_out_reduced = np.max(lens_depth_out_reduced,axis=1)


    # auroc

    fpr, tpr, thresholds = metrics.roc_curve(np.concatenate((np.ones(len(LD_in_reduced)),\
                            np.zeros(len(LD_out_reduced)))), np.concatenate((LD_in_reduced,LD_out_reduced)).reshape(-1,))
    auc = metrics.auc(fpr, tpr)
    
    auc_LD += [auc]
     
    
    print(datetime.now()-now)
print(datetime.now())

np.save('./auroc_LD_MNIST_hold_one_out_30_epochs_500_points.npy',np.array(auc_LD))

auc_LD2 = np.load('./auroc_LD_MNIST_hold_one_out_30_epochs_500_points.npy')
print(auc_LD2, np.mean(auc_LD2), np.std(auc_LD2))