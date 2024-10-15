# Statistical Depth Using Fermat Distance for Uncertainty Estimation

This Repo contains all the files related for computing Lens Depth and for conducting experiments in the related paper.

## Dependencies

The code is implemented based mainly on python library Pytorch (and torchvision). All needed libraries can be found in  [requirements.txt](https://github.com/HaiVyNGUYEN/LD/blob/master/requirements.txt). The code is supposed to be run in Linux but can be easily adapted for other systems. We strongly recommend to create virtual environment for a proper running (such as conda virtual env). This can be easily done in linux terminal as follow:
```
conda create -n yourenvname python=x.x anaconda
```
Then, to activate this virtual env:
```
conda activate yourenvname
```
To install a package in this virtual env:
```
conda install -n yourenvname [package]
```

To quit this env:

```
conda deactivate
```

## Data

In this work, we use only publicly available datasets [MNIST](https://en.wikipedia.org/wiki/MNIST_database), [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist), [SVHN](http://ufldl.stanford.edu/housenumbers/) and [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html). All these datasets can be downloaded using standard Deep Learning library [Pytorch](https://pytorch.org/).
Besides, for toydataset and proof-of-concept, we simulate dataset using python library [scikit-learn](https://scikit-learn.org/). 

## Running

To make it easier for practitioners, we intentionally make three 3 indepent directories:
1. [background_toy_dataset](https://github.com/HaiVyNGUYEN/LD/tree/master/background_toy_dataset): all the codes are in notebooks, so that we can demonstrate how the method works step by step. Besides, this aims to make sure that the method work correctly in 2D space with visual results.
2. [MNIST_FashionMNIST](https://github.com/HaiVyNGUYEN/LD/tree/master/MNIST_FashionMNIST): all the codes for conducting experiments related to MNIST/FashionMNIST.
3. [CIFAR10_SVHN](https://github.com/HaiVyNGUYEN/LD/tree/master/CIFAR10_SVHN): same for CFAR10/SVHN.

More detailed instructions can be found in README for each subdirectory. The codes in each of these directories can be runned independently. However, we strongly recommend to start by [background_toy_dataset](https://github.com/HaiVyNGUYEN/LD/tree/master/background_toy_dataset) for a good understanding of method as well as to know how to correctly use our method.

![Alt text](https://github.com/HaiVyNGUYEN/ld_official/blob/master/images/gauss_LD.png  "Motivation example of R2 space where 2 clusters are in form of two-moons. GDA (left) based on Gaussian assumption fails completely to capture the distribution of dataset whereas our proposed method (right) represents very well how central a point is w.r.t clusters")
![Alt text](https://github.com/HaiVyNGUYEN/ld_official/blob/master/images/two_moon.png "Different method for uncertainty estimation applied on a neural net trained to classify 2 classes in moon-shape (represented by yellow and black points respectively). Uncertainty estimations are computed based solely on the features space of the net without seeing directly the inputs.")

