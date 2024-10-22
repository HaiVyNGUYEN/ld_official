import sys
import os
current = os.path.dirname(os.path.realpath('./'))
sys.path.append(current)

from utils.ld_tools import *
from model_archi.model_archi_resnet18_SN import resnet18
from model_archi.model_archi_resnet18 import ResNet18_3
from model_archi.model_archi_wideresnet_SN import wrn

import random 
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split, Dataset
from sklearn.cluster import KMeans
from datetime import datetime
from sklearn import metrics
from tinyimagenet import TinyImageNet
from pathlib import Path

data_dir = 'dataset'
train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=False)
test_dataset  = torchvision.datasets.CIFAR10(data_dir, train=False, download=False)
test_dataset_out  = torchvision.datasets.SVHN(data_dir, split='test', download=False)

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
test_dataset_out.transform = test_transform

print(len(train_dataset),len(test_dataset), len(test_dataset_out))

batch_size= 128

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=False)
test_loader_out = torch.utils.data.DataLoader(test_dataset_out, batch_size=batch_size,shuffle=False)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
now = datetime.now()
for sim in range(1,6):
    
    print('Try: ',sim)
    # load model
    
    model = ResNet18_3().to(device)  #(feature_dim=25)
    model.load_state_dict(torch.load(f'./cifar10_resnet18_200_epochs_{sim}'))
    
#     model = resnet18(spectral_normalization=True).to(device)  #(feature_dim=512)
#     model.load_state_dict(torch.load(f'./cifar10_SN_resnet18_350_epochs_{sim}'))  
    
#     model = wrn(spectral_normalization=True).to(device)   #(feature_dim=640)
#     model.load_state_dict(torch.load(f'./cifar10_SN_wideresnet_350_epochs_{sim}'))

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
    
    feature_dim = 25
    #feature_dim = 512
    #feature_dim = 640

    # compuute feature of training set
    feature = np.zeros((10,5000,feature_dim))
    for i in range(10):
        print('Group: ',i)
        image_group = [u[0].cuda() for u in train_dataset if u[1]==i]
        image_group = torch.stack(image_group)
        print(image_group.shape)
        with torch.no_grad():
            for k in range(int(5000/500)):
                feature[i,k*500:k*500+500,:] = \
                np.array(model.forward_before_softmax(image_group[k*500:k*500+500]).detach().cpu()).reshape(-1,feature_dim)
    print(feature.shape)


    # K-mean
    print("K-mean")
    num_centers = 500
    centers = []
    for i in range(10):
        km = KMeans(n_clusters=num_centers,n_init=10)
        y_km = km.fit_predict(feature[i,:,:])
        c = np.array(km.cluster_centers_)
        print(c.shape)
        centers += [c]
    centers = np.array(centers)
    print(centers.shape)
    
    
    
    alpha=7

    # Internal fermat distance

    dist_mat_fermat_reduced = np.zeros((10,num_centers,num_centers))
    for i in range(10):
        print('Group:',i)
        dist_mat_group = distance_matrix(centers[i],centers[i])
        dist_mat_fermat_reduced[i] = fermat_function(dist_mat_group,alpha=alpha)


    # LD

    lens_depth_in_reduced = np.zeros((len(feature_test_in),10))

    # in

    batch_size = 250
    num_batches_in = int(len(feature_test_in)/batch_size)
    for i in range(10):
        dist_mat =  distance_matrix(centers[i],feature_test_in)
        print('Group',i)
        for k in range(num_batches_in):
            print('Processing batch:',k)
            dist_ext = fermat_function_ext_simultaneous(dist_mat_fermat_reduced[i],\
                                                        dist_mat[:,batch_size*k:batch_size*(k+1)].T,alpha=alpha)
            lens_depth_in_reduced[batch_size*k:batch_size*(k+1),i] = \
                                lens_depth_ext_function_simultaneous(dist_mat_fermat_reduced[i], dist_ext)

    # out

    lens_depth_out_reduced = np.zeros((len(feature_test_out),10))
    num_batches_out = int(len(feature_test_out)/batch_size)

    for i in range(10):
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

    np.save(f'./auroc_svhn_resnet_200_epochs_500_points_{sim}.npy',np.array([auc]))
    #np.save(f'./auroc_svhn_resnet_SN_350_epochs_500_points_{sim}.npy',np.array([auc]))
    #np.save(f'./auroc_svhn_wideresnet_SN_350_epochs_500_points_{sim}.npy',np.array([auc]))
   
    
    
    print(datetime.now()-now)
print(datetime.now())