import torchvision


data_dir = 'dataset'

# download CIFAR10 dataset

train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)


# download test set  of SVHN dataset

test_dataset_svhn  = torchvision.datasets.SVHN(data_dir, split='test', download=True)

# snip code for calculating mean and std

# x = np.concatenate([np.asarray(train_dataset[i][0])/255. for i in range(len(train_dataset))])
# print(x.shape)

# # calculate the mean and std along the (0, 1) axes
# train_mean = np.mean(x, axis=(0, 1))
# train_std = np.std(x, axis=(0, 1))
# # the the mean and std
# print(train_mean, train_std)
# # [0.49139968 0.48215841 0.44653091] [0.24703223 0.24348513 0.26158784]
# del x