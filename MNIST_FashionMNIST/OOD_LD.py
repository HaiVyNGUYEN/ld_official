import sys
import os
current = os.path.dirname(os.path.realpath('./'))
sys.path.append(current)
from utils.ld_tools import *
from model_archi.fashion_mnist_archi import MyOwnNeuralNetwork

import random
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, normalize
from datetime import datetime
from sklearn.cluster import KMeans


data_dir = 'dataset'

# Get cpu or gpu device.
device = "cuda" if torch.cuda.is_available() else "cpu"

# load fashionMNIST


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
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128,shuffle=True)


# load only the test set of MNIST

test_dataset_mnist  = torchvision.datasets.MNIST(data_dir, train=False, download=False)
test_dataset_mnist.transform = test_transform
test_loader_mnist = torch.utils.data.DataLoader(test_dataset_mnist, batch_size=128,shuffle=False)

# auroc score for deep nearest neighbors method

now = datetime.now()
for sim in range(1,6):
  
    model = MyOwnNeuralNetwork().to(device)
    model.load_state_dict(torch.load(f'./fashion_mnist_net_30_epochs_{sim}'))

    model.eval()

    # compute feature to test
    feature_test_in = None
    with torch.no_grad():
        for loader in test_loader:
            X,_ = loader
            X = X.cuda()
            array = np.array(model.forward_before_softmax(X).cpu())
            array = normalize(array, axis=1)
            if feature_test_in is None:
                feature_test_in = array
            else:
                feature_test_in = np.concatenate((feature_test_in,array))
    print(feature_test_in.shape)

    feature_test_out = None
    with torch.no_grad():
        for loader in test_loader_mnist:
            X,_ = loader
            X = X.cuda()
            array = np.array(model.forward_before_softmax(X).cpu())
            array = normalize(array, axis=1)
            if feature_test_out is None:
                feature_test_out = array
            else:
                feature_test_out = np.concatenate((feature_test_out,array))
    print(feature_test_out.shape)

    # compuute feature of training set
    feature = np.zeros((10,6000,25))
    for i in range(10):
        print('Group: ',i)
        image_group = [u[0].cuda() for u in train_dataset if u[1]==i]
        image_group = torch.stack(image_group)
        print(image_group.shape)
        with torch.no_grad():
            feature[i,:,:] = normalize(np.array(model.forward_before_softmax(image_group).detach().cpu()).reshape(-1,25),\
                                   axis=1)

    print(feature.shape)


    # K-mean
    print("K-mean")
    num_centers = 1500
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

    batch_size = 125
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

    np.save(f'./auroc_30_epochs_1500_points_no_augment_{sim}.npy',np.array([auc]))
    
    # rejection test

    LD = np.concatenate((LD_in_reduced,LD_out_reduced))
    quantiles = []
    for i in np.arange(1,10)*0.1:
        quantiles += [np.quantile(LD,i)]


    acc_confidence = []
    correct_in = accuracy_evaluation(test_loader,model)*len(test_dataset)
    acc_total = correct_in/(len(test_dataset)+ len(test_dataset_mnist))
    acc_confidence += [acc_total]
    for i,q in enumerate(quantiles):
        print('Quantile', (i+1)/10)
        rest_test_data = [test_dataset[i] for i in np.where(LD_in_reduced>q)[0]]
        rest_test_loader = torch.utils.data.DataLoader(rest_test_data, batch_size=128,shuffle=False)
        correct_in = accuracy_evaluation(rest_test_loader,model)*len(rest_test_data)
        no_out = len(np.where(LD_out_reduced>q)[0])
        acc_total = correct_in/(len(rest_test_data)+no_out)
        acc_confidence += [acc_total]

    np.save(f'./acc_confidence_30_epochs_1500_points_no_augment_{sim}.npy',acc_confidence)
    
    
    print(datetime.now()-now)
print(datetime.now())