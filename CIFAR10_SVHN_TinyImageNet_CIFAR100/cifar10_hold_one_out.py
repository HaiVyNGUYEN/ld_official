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
from torch.optim.lr_scheduler import MultiStepLR

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import math
from sklearn.cluster import KMeans



data_dir = 'dataset'
train_dataset0 = torchvision.datasets.CIFAR10(data_dir, train=True, download=False)
test_dataset0  = torchvision.datasets.CIFAR10(data_dir, train=False, download=False)

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

train_dataset0.transform = train_transform
test_dataset0.transform = test_transform

train_dataset = Subset(train_dataset0, np.where(np.array([u[1] for u in train_dataset0])!=9)[0])
test_dataset = Subset(test_dataset0, np.where(np.array([u[1] for u in test_dataset0])!=9)[0])
hold_out_test_dataset = Subset(test_dataset0, np.where(np.array([u[1] for u in test_dataset0])==9)[0])

print(len(train_dataset),len(test_dataset), len(hold_out_test_dataset))

batch_size= 128

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
test_loader_out = torch.utils.data.DataLoader(hold_out_test_dataset, batch_size=batch_size,shuffle=False)

### Defining model


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock3(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out





class ResNet3(nn.Module):
    def __init__(self, block, num_blocks, num_classes=9,inter_dim=25):
        super(ResNet3, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(512*block.expansion, inter_dim)
        self.linear2 = nn.Linear(inter_dim,num_classes)
        self.gap = torch.nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_before_softmax(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out
    
    def forward(self,x):
        out = self.forward_before_softmax(x)
        out = self.linear2(out)
        return out


def ResNet18_3(num_classes=9,inter_dim=25):
    return ResNet3(BasicBlock3, [2,2,2,2], num_classes,inter_dim)
    
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
    model = ResNet18_3().to(device)
    #print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                    momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    loss_fn = nn.CrossEntropyLoss()

    train_batch_loss = []
    train_batch_accuracy = []
    valid_batch_accuracy = []
    valid_batch_loss = []
    train_epoch_no = []
    valid_epoch_no = []
    epochs = 200
    from datetime import datetime
    now = datetime.now()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        _train_batch_loss , _train_batch_accuracy = train(train_loader, model, loss_fn, optimizer)
        _valid_batch_loss , _valid_batch_accuracy, correct_temp = validation(test_loader, model, loss_fn)

        if (t+1)%5==0:
            print(datetime.now()-now)

        scheduler.step()

    torch.save(model.state_dict(), f'./cifar10_hold_one_out_resnet18_200_epochs_{i}')

    print(datetime.now())

print("Done!")


### LD method

now = datetime.now()
auc_LD = []
for sim in range(1,6):
    
    print('Try:', sim)
    # load model
    
    model = ResNet18_3().to(device)
    model.load_state_dict(torch.load(f'./cifar10_hold_one_out_resnet18_200_epochs_{sim}'))
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
    feature = np.zeros((9,5000,25))
    for i in range(9):
        print('Group: ',i)
        image_group = [u[0].cuda() for u in train_dataset if u[1]==i]
        image_group = torch.stack(image_group)
        print(image_group.shape)
        with torch.no_grad():
            feature[i,:,:] = np.array(model.forward_before_softmax(image_group).detach().cpu()).reshape(-1,25)

    print(feature.shape)


    # K-mean
    print("K-mean")
    num_centers = 500
    centers = []
    for i in range(9):
        km = KMeans(n_clusters=num_centers,n_init=10)
        y_km = km.fit_predict(feature[i,:,:])
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

np.save('./auroc_LD_cifar10_hold_one_out_200_epochs_500_points.npy',np.array(auc_LD))