import sys
import os
current = os.path.dirname(os.path.realpath('./'))
sys.path.append(current)

from utils.ld_tools import *
from utils.training_tools import *
from model_archi.model_archi_resnet18 import ResNet18_3

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


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
## Defining optimizer and loss functions 
for i in range(1,6):
    print(('Try ',i))
    model = ResNet18_3(num_classes=9).to(device)
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
        _train_batch_loss , _train_batch_accuracy = train(train_loader, model, loss_fn, optimizer, device)
        _valid_batch_loss , _valid_batch_accuracy, correct_temp = validation(test_loader, model, loss_fn, device)

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
    
    model = ResNet18_3(num_classes=9).to(device)
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