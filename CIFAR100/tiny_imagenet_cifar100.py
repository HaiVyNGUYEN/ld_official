import sys
import os
current = os.path.dirname(os.path.realpath('./'))
sys.path.append(current)
from utils.ld_tools import *
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn import metrics
from tinyimagenet import TinyImageNet
from pathlib import Path
from model_archi_wideresnet_SN import wrn



data_dir = 'dataset'
train_dataset = torchvision.datasets.CIFAR100(data_dir, train=True, download=False)
test_dataset  = torchvision.datasets.CIFAR100(data_dir, train=False, download=False)
test_dataset_out = TinyImageNet(Path('./tiny_imagenet/'),split='test')

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

test_transform_out = transforms.Compose([
                               transforms.ToTensor(), transforms.Resize(32),
                                transforms.Normalize(
                                (0.49139968, 0.48215841, 0.44653091), (0.2023, 0.1994, 0.2010))])

train_dataset.transform = train_transform
test_dataset.transform = test_transform
test_dataset_out.transform = test_transform_out

print(len(train_dataset),len(test_dataset), len(test_dataset_out))

batch_size= 128

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
test_loader_out = torch.utils.data.DataLoader(test_dataset_out, batch_size=batch_size,shuffle=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

now = datetime.now()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
now = datetime.now()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

now = datetime.now()

for sim in range(1,6):
    # load model   
    
    model = wrn(spectral_normalization=True,num_classes=100).to(device)
    model.load_state_dict(torch.load(f'./cifar100_SN_wideresnet_350_epochs_{sim}'))

    model.eval()
    
    
    feature_dim = 640  #(for wideresnet)
    
    
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
    feature = np.zeros((100,500,feature_dim))  
    for i in range(100):
        print('Group: ',i)
        image_group = [u[0].cuda() for u in train_dataset if u[1]==i]
        image_group = torch.stack(image_group)
        print(image_group.shape)  # (5000, 3, 32, 32)
        process_batchsize = 250
        with torch.no_grad():
            #print(image_group.shape[0]/process_batchsize)
            for k in range(int(image_group.shape[0]/process_batchsize)):
                feature[i,k*process_batchsize:(k+1)*process_batchsize,:] = \
                np.array(model.forward_before_softmax(image_group[k*process_batchsize:(k+1)*process_batchsize]).cpu())

    print(feature.shape)


    # K-mean
    
    num_centers = 500
    
    
    alpha=7

    # Internal fermat distance

    dist_mat_fermat_reduced = np.zeros((100,num_centers,num_centers))
    for i in range(100):
        print('Group:',i)
        dist_mat_group = distance_matrix(feature[i],feature[i])
        dist_mat_fermat_reduced[i] = fermat_function(dist_mat_group,alpha=alpha)


    # LD

    lens_depth_in_reduced = np.zeros((len(feature_test_in),100))

    # in

    batch_size = 250
    num_batches_in = int(len(feature_test_in)/batch_size)
    for i in range(100):
        dist_mat =  distance_matrix(feature[i],feature_test_in)
        print('Group',i)
        for k in range(num_batches_in):
            print('Processing batch:',k)
            dist_ext = fermat_function_ext_simultaneous(dist_mat_fermat_reduced[i],\
                                                        dist_mat[:,batch_size*k:batch_size*(k+1)].T,alpha=alpha)
            lens_depth_in_reduced[batch_size*k:batch_size*(k+1),i] = \
                                lens_depth_ext_function_simultaneous(dist_mat_fermat_reduced[i], dist_ext)

    # out

    lens_depth_out_reduced = np.zeros((len(feature_test_out),100))
    num_batches_out = int(len(feature_test_out)/batch_size)

    for i in range(100):
        dist_mat =  distance_matrix(feature[i],feature_test_out)
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
    
    print(auc)

    np.save(f'./auroc_wideresnet_tiny_imagenet_350_epochs_500_points_{sim}.npy',np.array([auc]))

    
    print(datetime.now()-now)
print(datetime.now())