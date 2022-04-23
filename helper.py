import scipy.stats
import scipy.io
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms
from sklearn.metrics import auc, roc_curve
import poison


if torch.cuda.is_available():
    device = torch.device('cuda')
    CUDA = True
else:
    device = torch.device('cpu')
    CUDA = False

# You can modify these parameters for dataloader according to your environments.
kwargs = {'num_workers': 8, 'pin_memory': True} if CUDA else {}
# This is the root path of all the datasets. Mkdir a dataset named 'datasets' under your project folder.
Root = 'datasets/'


normalize_cifar = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

normal_transform = transforms.ToTensor()
# normal_transform = transforms.Compose([
#     transforms.ToTensor(),
#     normalize_cifar])

# Black and white images apply simple augmentation.
augment_transform_bw = transforms.Compose([
    transforms.RandomCrop(28, padding=2),
    transforms.ToTensor()])
augment_transform_bw = transforms.RandomChoice([augment_transform_bw, normal_transform])

# Colorful images apply random flip and crop augmentation.
augment_transform_color = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()])
augment_transform_color = transforms.RandomChoice([augment_transform_color, normal_transform])

# SVHN doesn't apply random horizontal flip
augment_transform_svhn = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()])
augment_transform_svhn = transforms.RandomChoice([augment_transform_svhn, normal_transform])


# Generate dataloaders of different datasets.
def get_data(dataname, batch_size, use_augment=True, use_split=True, split=10000):

    if dataname in ['MNIST', 'FashionMNIST']:
        train_transform = augment_transform_bw if use_augment else normal_transform

    if dataname in ['CIFAR5', 'CIFAR10', 'CIFAR100', 'LSUN']:
        train_transform = augment_transform_color if use_augment else normal_transform

    if dataname in ['SVHN']:
        train_transform = augment_transform_svhn if use_augment else normal_transform


    if dataname == 'SVHN':
        dataset_kwargs = [{'split': 'train'}, {'split': 'train'}, {'split': 'test'}]
    elif dataname == 'LSUN':
        dataset_kwargs = [{'classes': 'train'}, {'classes': 'val'}, {'classes': 'test'}]
    elif dataname in ['CIFAR5', 'CIFAR10', 'CIFAR100', 'FashionMNIST', 'MNIST']:
        dataset_kwargs = [{'train': True}, {'train': True}, {'train': False}]

    train_data = getattr(datasets, dataname)(Root + str(dataname), transform=train_transform, download=True, **dataset_kwargs[0])
    valid_data = getattr(datasets, dataname)(Root + str(dataname), transform=normal_transform, download=True, **dataset_kwargs[1])
    test_data = getattr(datasets, dataname)(Root + str(dataname), transform=normal_transform, download=True, **dataset_kwargs[2])
    print(len(train_data), len(valid_data), len(test_data))

    if use_split:
        train_num = len(train_data) - split
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_num, split])
    #
    # kwargs = {'num_workers': 16, 'pin_memory': True} if CUDA else {}  #########TODO

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,  shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,  shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader

# Generate dataloaders of backdoored datasets.
def get_backdoored_mnist(dataname, batch_size, poison_rate, use_augment=True, use_split=True, split=10000):
    if dataname in ['MNIST', 'FashionMNIST']:
        train_transform = augment_transform_bw if use_augment else normal_transform

    if dataname in ['CIFAR5', 'CIFAR10', 'CIFAR100', 'LSUN']:
        train_transform = augment_transform_color if use_augment else normal_transform

    if dataname in ['SVHN']:
        train_transform = augment_transform_svhn if use_augment else normal_transform
    if dataname == 'SVHN':
        dataset_kwargs = [{'split': 'train'}, {'split': 'train'}, {'split': 'test'}]
    elif dataname == 'LSUN':
        dataset_kwargs = [{'classes': 'train'}, {'classes': 'val'}, {'classes': 'test'}]
    elif dataname in ['CIFAR5', 'CIFAR10', 'CIFAR100', 'FashionMNIST', 'MNIST']:
        dataset_kwargs = [{'train': True}, {'train': True}, {'train': False}]
    train_data = getattr(datasets, dataname)(Root + str(dataname), transform=train_transform, download=True, **dataset_kwargs[0])
    valid_data = getattr(datasets, dataname)(Root + str(dataname), transform=normal_transform, download=True, **dataset_kwargs[1])
    test_data = getattr(datasets, dataname)(Root + str(dataname), transform=normal_transform, download=True, **dataset_kwargs[2])
    print(len(train_data), len(valid_data), len(test_data))


    poison_type = 'badnets'
    poison_target = 0
    trigger_alpha = 1.0
    

    train_data.data = np.array(train_data.data[:,:,:,np.newaxis])
    train_data.targets = np.array(train_data.targets)
    test_data.data = np.array(test_data.data[:,:,:,np.newaxis])
    test_data.targets = np.array(test_data.targets)
    triggers = {'badnets': 'checkerboard_1corner',
                'clean-label': 'checkerboard_4corner',
                'blend': 'gaussian_noise',
                'benign': None}
    trigger_type = triggers[poison_type]
    if poison_type in ['badnets', 'blend']:
        poison_train, trigger_info = \
            poison.add_trigger(data_set=train_data, trigger_type=trigger_type, poison_rate=poison_rate,
                                     poison_target=poison_target, trigger_alpha=trigger_alpha, s=28)
        poison_test = poison.add_predefined_trigger(data_set=test_data, trigger_info=trigger_info)
    poison_train.data = torch.from_numpy(poison_train.data[:,:,:,0])
    poison_train.targets = torch.from_numpy(poison_train.targets)
    poison_test.data = torch.from_numpy(poison_test.data[:,:,:,0])
    poison_test.targets = torch.from_numpy(poison_test.targets)
    test_data.data = torch.from_numpy(test_data.data[:,:,:,0])
    test_data.targets = torch.from_numpy(test_data.targets)


    poison_train_loader = torch.utils.data.DataLoader(poison_train, batch_size=batch_size, shuffle=True, **kwargs)
    poison_test_loader = torch.utils.data.DataLoader(poison_test, batch_size=batch_size, **kwargs)
    clean_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, **kwargs)

    return poison_train_loader, poison_test_loader, clean_test_loader

# Generate dataloaders of backdoored datasets.
def get_backdoored_cifar(dataname, batch_size, use_augment=True, use_split=True, split=10000):
    if dataname in ['MNIST', 'FashionMNIST']:
        train_transform = augment_transform_bw if use_augment else normal_transform

    if dataname in ['CIFAR5', 'CIFAR10', 'CIFAR100', 'LSUN']:
        train_transform = augment_transform_color if use_augment else normal_transform

    if dataname in ['SVHN']:
        train_transform = augment_transform_svhn if use_augment else normal_transform
    if dataname == 'SVHN':
        dataset_kwargs = [{'split': 'train'}, {'split': 'train'}, {'split': 'test'}]
    elif dataname == 'LSUN':
        dataset_kwargs = [{'classes': 'train'}, {'classes': 'val'}, {'classes': 'test'}]
    elif dataname in ['CIFAR5', 'CIFAR10', 'CIFAR100', 'FashionMNIST', 'MNIST']:
        dataset_kwargs = [{'train': True}, {'train': True}, {'train': False}]
    train_data = getattr(datasets, dataname)(Root + str(dataname), transform=train_transform, download=True, **dataset_kwargs[0])
    valid_data = getattr(datasets, dataname)(Root + str(dataname), transform=normal_transform, download=True, **dataset_kwargs[1])
    test_data = getattr(datasets, dataname)(Root + str(dataname), transform=normal_transform, download=True, **dataset_kwargs[2])
    print(len(train_data), len(valid_data), len(test_data))

    import cv2
    cv2.imwrite('./vis00.png',test_data.data[0])
    cv2.imwrite('./vis01.png',test_data.data[1])

    poison_type = 'badnets'
    poison_target = 0
    poison_rate = 0.05
    trigger_alpha = 0.2

    triggers = {'badnets': 'checkerboard_1corner',
                'clean-label': 'checkerboard_4corner',
                'blend': 'gaussian_noise',
                'benign': None}
    trigger_type = triggers[poison_type]

    if poison_type in ['badnets', 'blend']:
        poison_train, trigger_info = \
            poison.add_trigger(data_set=train_data, trigger_type=trigger_type, poison_rate=poison_rate,
                                     poison_target=poison_target, trigger_alpha=trigger_alpha, s=32)
        poison_test = poison.add_predefined_trigger(data_set=test_data, trigger_info=trigger_info)

    cv2.imwrite('./vis10.png',poison_test.data[0])
    cv2.imwrite('./vis11.png',poison_test.data[1])

    poison_train_loader = torch.utils.data.DataLoader(poison_train, batch_size=batch_size, shuffle=True, **kwargs)
    poison_test_loader = torch.utils.data.DataLoader(poison_test, batch_size=batch_size, **kwargs)
    clean_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, **kwargs)

    return poison_train_loader, poison_test_loader, clean_test_loader


# This function converts the first 0 - 4 class in CIFAR-10 as in-distribution data.
#  and the last five classes 5-9 as out-of-distribution data.

def get_cifar5(batch_size, split, use_augment, validation):
    dataname = 'CIFAR10'
    train_transform = augment_transform_color if use_augment else normal_transform

    train_data = getattr(datasets, dataname) \
        (Root + str(dataname), train=True, download=True, transform=train_transform)

    test_data = getattr(datasets, dataname)(Root + str(dataname), train=False, download=True,
                                            transform=normal_transform)

    train_img, train_targets = np.array(train_data.data), np.array(train_data.targets)
    # valid_img, valid_targets = np.array(valid_data.data), np.array(valid_data.targets)
    test_img, test_targets = np.array(test_data.data), np.array(test_data.targets)

    # 0~ 4 in class train and valid
    train_in_idx = np.where(train_targets < 5)[0]  # training in-class index

    if validation:
        indices = train_in_idx
        np.random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]
    else:
        # split = 0
        train_idx = train_in_idx

    # 0~4 in class test data
    test_in_idx = np.where(test_targets < 5)[0]
    # 5~9 ood
    test_ood_idx = np.where(test_targets >= 5)[0]

    # kwargs = {'num_workers': 8, 'pin_memory': True} if CUDA else {}  #####

    train_loader = torch.utils.data.DataLoader(Subset(train_data, train_idx), batch_size=batch_size,
                                               shuffle=True, **kwargs)
    valid_loader = torch.utils.data.DataLoader(Subset(train_data, valid_idx), batch_size=batch_size,
                                               shuffle=False, **kwargs) if validation else None
    test_loader = torch.utils.data.DataLoader(Subset(test_data, test_in_idx), batch_size=batch_size,
                                              shuffle=False, **kwargs)
    ood_loader = torch.utils.data.DataLoader(Subset(test_data, test_ood_idx), batch_size=batch_size,
                                             shuffle=False, **kwargs)
    return train_loader, valid_loader, test_loader, ood_loader


# Generate not-mnist dataloader. An OOD dataset for evaluating FashionMNIST.
# Please download the dataset in http://yaroslavvb.com/upload/notMNIST/

def get_notmnist(batch_size, split):
    dataset = scipy.io.loadmat(Root + "notMNIST_small.mat")
    data = np.array(dataset["images"]).transpose() / 255
    num = len(data)
    indices = list(range(num))
    np.random.shuffle(indices)
    data = data[indices[:split]]
    data = torch.tensor(data)
    data = data.unsqueeze(1).float()
    notmnist_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, **kwargs)
    return notmnist_loader

# For LSUN, you can use LSUN_resize or LSUN_classroom
# This function is used to read and generate LSUN_resize.
# Please download LSUN_resize in https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz

def get_lsun(batch_size, split):
    data = datasets.ImageFolder(root= Root + 'LSUN_resize',
                         transform=normal_transform)
    num = len(data)
    indices = list(range(num))
    np.random.shuffle(indices)
    split_idx = indices[:split]
    lsun_loader = torch.utils.data.DataLoader(Subset(data, split_idx), batch_size=batch_size, **kwargs)
    return lsun_loader


# Calculate dissonance of a vector of alpha #
def getDisn(alpha):
    evi = alpha - 1
    s = torch.sum(alpha, axis=1, keepdims=True)
    blf = evi / s
    idx = np.arange(alpha.shape[1])
    diss = 0
    Bal = lambda bi, bj: 1 - torch.abs(bi - bj) / (bi + bj + 1e-8)
    for i in idx:
        score_j_bal = [blf[:, j] * Bal(blf[:, j], blf[:, i]) for j in idx[idx != i]]
        score_j = [blf[:, j] for j in idx[idx != i]]
        diss += blf[:, i] * sum(score_j_bal) / (sum(score_j) + 1e-8)
    return diss


# Calculate entropy of a vector of probability
def cal_entropy(p):
    if type(p) == torch.Tensor:
        return (-p * torch.log(p + 1e-8)).sum(1)
    else:
        return (-p * np.log(p + 1e-8)).sum(1)


# Evaluation:  get roc curve from a set of normal scores and anormal scores.
def get_pr_roc(normal_score, anormal_score):

    if type(normal_score) == pd.core.series.Series:
        normal_score = normal_score.iloc[0]
    if type(anormal_score) == pd.core.series.Series:
        anormal_score = anormal_score.iloc[0]

    truth = np.zeros((len(normal_score) + len(anormal_score)))
    truth[len(normal_score):] = 1

    score = np.concatenate([normal_score, anormal_score])

    fpr, tpr, _ = roc_curve(truth, score, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


# def fgsm_attack(image, epsilon, data_grad):
#     # Collect the element-wise sign of the data gradient
#     sign_data_grad = data_grad.sign()
#     # Create the perturbed image by adjusting each pixel of the input image
#     perturbed_image = image + epsilon*sign_data_grad
#     # Adding clipping to maintain [0,1] range
#     perturbed_image = torch.clamp(perturbed_image, 0, 1)
#     # Return the perturbed image
#     return perturbed_image
