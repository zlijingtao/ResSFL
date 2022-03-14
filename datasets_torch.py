# import tensorflow as tf
import numpy as np
from torchvision.transforms.transforms import CenterCrop
import tqdm
import torch.nn.functional as F
# import sklearn
import matplotlib.pyplot as plt
import torch
# import tensorflow_datasets as tfds
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, sampler, DataLoader
import urllib
import tarfile
import os
BUFFER_SIZE = 10000
SIZE = 32

# getImagesDS = lambda X, n: np.concatenate([x[0].numpy()[None,] for x in X.take(n)])
CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.247, 0.243, 0.261)
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
FACESCRUB_TRAIN_MEAN = (0.5708, 0.5905, 0.4272)
FACESCRUB_TRAIN_STD = (0.2058, 0.2275, 0.2098)
TINYIMAGENET_TRAIN_MEAN = (0.5141, 0.5775, 0.3985)
TINYIMAGENET_TRAIN_STD = (0.2927, 0.2570, 0.1434)
SVHN_TRAIN_MEAN = (0.3522, 0.4004, 0.4463)
SVHN_TRAIN_STD = (0.1189, 0.1377, 0.1784)

def getImagesDS(X, n):
    image_list = []
    for i in range(n):
        image_list.append(X[i][0].numpy()[None,])
    return np.concatenate(image_list)

class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        images, labels = self.dataset[self.idxs[item]]
        return images, labels

def remove_class_loader(some_dataset, label_class, batch_size=16, num_workers=2):
    def remove_one_label(target, label):
        label_indices = []
        excluded_indices = []
        for i in range(len(target)):
            if target[i] != label:
                label_indices.append(i)
            else:
                excluded_indices.append(i)
        return label_indices, excluded_indices



    indices, excluded_indices = remove_one_label(some_dataset.targets, label_class)

    new_data_loader = DataLoader(
        some_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, sampler = torch.utils.data.sampler.SubsetRandomSampler(indices))
    excluded_data_loader = DataLoader(
        some_dataset, shuffle=False, num_workers=num_workers, batch_size=batch_size, sampler = torch.utils.data.sampler.SubsetRandomSampler(excluded_indices))
    
    return new_data_loader, excluded_data_loader


def noniid_unlabel(dataset, num_users, label_rate, noniid_ratio = 0.2, num_class = 10):
    num_class_per_client = int(noniid_ratio * num_class)
    num_shards, num_imgs = num_class_per_client * num_users, int(len(dataset)/num_users/num_class_per_client)
    idx_shard = [i for i in range(num_shards)]
    dict_users_unlabeled = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))  
    

    for i in range(len(dataset)):
        labels[i] = dataset[i][1]
        
    num_items = int(len(dataset)/num_users)
    dict_users_labeled = set()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]#索引值
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_class_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_unlabeled[i] = np.concatenate((dict_users_unlabeled[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)

    dict_users_labeled = set(np.random.choice(list(idxs), int(len(idxs) * label_rate), replace=False))
    
    for i in range(num_users):

        dict_users_unlabeled[i] = set(dict_users_unlabeled[i])
#         dict_users_labeled = dict_users_labeled | set(np.random.choice(list(dict_users_unlabeled[i]), int(num_items * label_rate), replace=False))
        dict_users_unlabeled[i] = dict_users_unlabeled[i] - dict_users_labeled


    return dict_users_labeled, dict_users_unlabeled

def noniid_alllabel(dataset, num_users, noniid_ratio = 0.2, num_class = 10):
    num_class_per_client = int(noniid_ratio * num_class)
    num_shards, num_imgs = num_class_per_client * num_users, int(len(dataset)/num_users/num_class_per_client)
    idx_shard = [i for i in range(num_shards)]
    dict_users_labeled = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(len(dataset))
    labels = np.arange(len(dataset))  
    

    for i in range(len(dataset)):
        labels[i] = dataset[i][1]
        
    num_items = int(len(dataset)/num_users)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]#索引值
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, num_class_per_client, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users_labeled[i] = np.concatenate((dict_users_labeled[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    
    for i in range(num_users):

        dict_users_labeled[i] = set(dict_users_labeled[i])


    return dict_users_labeled

def load_fmnist():
    xpriv = datasets.FashionMNIST(root='./data', train=True, download=True)

    xpub = datasets.FashionMNIST(root='./data', train=False)

    x_train = np.array(xpriv.data)
    y_train = np.array(xpriv.targets)
    x_test = np.array(xpub.data)
    y_test = np.array(xpub.targets)
    
    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]
    x_train = np.tile(x_train, (1,3,1,1))
    x_test = np.tile(x_test, (1,3,1,1))

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train).type(torch.LongTensor)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test).type(torch.LongTensor)
    x_train = F.interpolate(x_train, (32, 32))
    x_test = F.interpolate(x_test, (32, 32))
    x_train  = x_train / (255/2) - 1
    x_test  = x_test / (255/2) - 1
    x_train = torch.clip(x_train, -1., 1.)
    x_test = torch.clip(x_test, -1., 1.)
    # Need a different way to denormalize
    xpriv = TensorDataset(x_train, y_train)
    xpub = TensorDataset(x_test, y_test)
    return xpriv, xpub

def load_mnist():
    xpriv = datasets.MNIST(root='./data', train=True, download=True)

    xpub = datasets.MNIST(root='./data', train=False)

    x_train = np.array(xpriv.data)
    y_train = np.array(xpriv.targets)
    x_test = np.array(xpub.data)
    y_test = np.array(xpub.targets)
    
    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]
    x_train = np.tile(x_train, (1,3,1,1))
    x_test = np.tile(x_test, (1,3,1,1))

    x_train = torch.Tensor(x_train)
    y_train = torch.Tensor(y_train).type(torch.LongTensor)
    x_test = torch.Tensor(x_test)
    y_test = torch.Tensor(y_test).type(torch.LongTensor)
    x_train = F.interpolate(x_train, (32, 32))
    x_test = F.interpolate(x_test, (32, 32))
    x_train  = x_train / (255/2) - 1
    x_test  = x_test / (255/2) - 1
    x_train = torch.clip(x_train, -1., 1.)
    x_test = torch.clip(x_test, -1., 1.)
    # Need a different way to denormalize
    xpriv = TensorDataset(x_train, y_train)
    xpub = TensorDataset(x_test, y_test)
    return xpriv, xpub

def get_mnist_bothloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, collude_use_public = False):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    mnist_training, mnist_testing = load_mnist()
    
    if num_client == 1:
        mnist_training_loader = [torch.utils.data.DataLoader(mnist_training,  batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers)]
    elif num_client > 1:
        mnist_training_loader = []
        for i in range(num_client):
            mnist_training_subset = torch.utils.data.Subset(mnist_training, list(range(i * (len(mnist_training)//num_client), (i+1) * (len(mnist_training)//num_client))))
            subset_training_loader = DataLoader(
                mnist_training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            mnist_training_loader.append(subset_training_loader)
    
    mnist_testing_loader = torch.utils.data.DataLoader(mnist_testing,  batch_size=batch_size, shuffle=False,
                num_workers=num_workers)

    return mnist_training_loader, mnist_testing_loader

def get_fmnist_bothloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, collude_use_public = False):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    fmnist_training, fmnist_testing = load_fmnist()
    
    if num_client == 1:
        fmnist_training_loader = [torch.utils.data.DataLoader(fmnist_training,  batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers)]
    elif num_client > 1:
        fmnist_training_loader = []
        for i in range(num_client):
            fmnist_training_subset = torch.utils.data.Subset(fmnist_training, list(range(i * (len(fmnist_training)//num_client), (i+1) * (len(fmnist_training)//num_client))))
            subset_training_loader = DataLoader(
                fmnist_training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            fmnist_training_loader.append(subset_training_loader)
    
    fmnist_testing_loader = torch.utils.data.DataLoader(fmnist_testing,  batch_size=batch_size, shuffle=False,
                num_workers=num_workers)

    return fmnist_training_loader, fmnist_testing_loader

def get_facescrub_bothloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, collude_use_public = False):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(FACESCRUB_TRAIN_MEAN, FACESCRUB_TRAIN_STD)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FACESCRUB_TRAIN_MEAN, FACESCRUB_TRAIN_STD)
    ])

    if not os.path.isdir("./facescrub-dataset/32x32/train"):
        os.system("git clone https://github.com/theothings/facescrub-dataset.git")
        import subprocess
        subprocess.call("python prepare_facescrub.py", shell=True)
    facescrub_training = datasets.ImageFolder('facescrub-dataset/32x32/train', transform=transform_train)
    facescrub_testing = datasets.ImageFolder('facescrub-dataset/32x32/validate', transform=transform_test)
    if num_client == 1:
        facescrub_training_loader = [torch.utils.data.DataLoader(facescrub_training,  batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers)]
    elif num_client > 1:
        facescrub_training_loader = []
        for i in range(num_client):
            mnist_training_subset = torch.utils.data.Subset(facescrub_training, list(range(i * (len(facescrub_training)//num_client), (i+1) * (len(facescrub_training)//num_client))))
            subset_training_loader = DataLoader(
                mnist_training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            facescrub_training_loader.append(subset_training_loader)
    
    facescrub_testing_loader = torch.utils.data.DataLoader(facescrub_testing,  batch_size=batch_size, shuffle=False,
                num_workers=num_workers)

    return facescrub_training_loader, facescrub_testing_loader

def get_tinyimagenet_bothloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, collude_use_public = False):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)
    ])

    if not os.path.isdir("./tiny-imagenet-200/train"):
        import subprocess
        subprocess.call("python prepare_tinyimagenet.py", shell=True)
    tinyimagenet_training = datasets.ImageFolder('tiny-imagenet-200/train', transform=transform_train)
    tinyimagenet_testing = datasets.ImageFolder('tiny-imagenet-200/val', transform=transform_test)

    if num_client == 1:
        tinyimagenet_training_loader = [torch.utils.data.DataLoader(tinyimagenet_training,  batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers)]
    elif num_client > 1:
        tinyimagenet_training_loader = []
        for i in range(num_client):
            mnist_training_subset = torch.utils.data.Subset(tinyimagenet_training, list(range(i * (len(tinyimagenet_training)//num_client), (i+1) * (len(tinyimagenet_training)//num_client))))
            subset_training_loader = DataLoader(
                mnist_training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            tinyimagenet_training_loader.append(subset_training_loader)
    
    tinyimagenet_testing_loader = torch.utils.data.DataLoader(tinyimagenet_testing,  batch_size=batch_size, shuffle=False,
                num_workers=num_workers)

    return tinyimagenet_training_loader, tinyimagenet_testing_loader

def get_purchase_trainloader():
    DATASET_PATH='./datasets/purchase'
    DATASET_NAME= 'dataset_purchase'

    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    DATASET_FILE = os.path.join(DATASET_PATH,DATASET_NAME)

    if not os.path.isfile(DATASET_FILE):
        print("Dowloading the dataset...")
        urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",os.path.join(DATASET_PATH,'tmp.tgz'))
        print('Dataset Dowloaded')

        tar = tarfile.open(os.path.join(DATASET_PATH,'tmp.tgz'))
        tar.extractall(path=DATASET_PATH)


    data_set =np.genfromtxt(DATASET_FILE,delimiter=',')

    X = data_set[:,1:].astype(np.float64)
    Y = (data_set[:,0]).astype(np.int32)-1

    len_train =len(X)
    r = np.load('./dataset_shuffle/random_r_purchase100.npy')
    X=X[r]
    Y=Y[r]
    train_classifier_ratio, train_attack_ratio = 0.1,0.15
    train_classifier_data = X[:int(train_classifier_ratio*len_train)]
    test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]

    train_classifier_label = Y[:int(train_classifier_ratio*len_train)]
    test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]

    xpriv = TensorDataset(train_classifier_data, train_classifier_label)
    xpub = TensorDataset(test_data, test_label)


    train_classifier_ratio, train_attack_ratio = 0.1,0.3
    train_data = X[:int(train_classifier_ratio*len_train)]
    test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    
    train_label = Y[:int(train_classifier_ratio*len_train)]
    test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]
    
    np.random.seed(100)
    train_len = train_data.shape[0]
    r = np.arange(train_len)
    np.random.shuffle(r)
    shadow_indices = r[:train_len//2]
    target_indices = r[train_len//2:]

    shadow_train_data, shadow_train_label = train_data[shadow_indices], train_label[shadow_indices]
    target_train_data, target_train_label = train_data[target_indices], train_label[target_indices]

    test_len = 1*train_len
    r = np.arange(test_len)
    np.random.shuffle(r)
    shadow_indices = r[:test_len//2]
    target_indices = r[test_len//2:]
    
    shadow_test_data, shadow_test_label = test_data[shadow_indices], test_label[shadow_indices]
    target_test_data, target_test_label = test_data[target_indices], test_label[target_indices]

    shadow_train = tensor_data_create(shadow_train_data, shadow_train_label)
    shadow_train_loader = DataLoader(shadow_train, batch_size=batch_size, shuffle=True, num_workers=1)

    shadow_test = tensor_data_create(shadow_test_data, shadow_test_label)
    shadow_test_loader = DataLoader(shadow_test, batch_size=batch_size, shuffle=True, num_workers=1)

    target_train = tensor_data_create(target_train_data, target_train_label)
    target_train_loader = DataLoader(target_train, batch_size=batch_size, shuffle=True, num_workers=1)

    target_test = tensor_data_create(target_test_data, target_test_label)
    target_test_loader = DataLoader(target_test, batch_size=batch_size, shuffle=True, num_workers=1)
    print('Data loading finished')
    return shadow_train_loader, shadow_test_loader, target_train_loader, target_test_loader

def get_cifar10_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, collude_use_public = False, data_portion = 1.0, noniid_ratio = 1.0):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
    ])
    #cifar00_training = CIFAR10Train(path, transform=transform_train)
    cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    indices = torch.randperm(len(cifar10_training))[:int(len(cifar10_training)* data_portion)]

    cifar10_training = torch.utils.data.Subset(cifar10_training, indices)

    if num_client == 1:
        cifar10_training_loader = [DataLoader(
            cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)]
    elif num_client > 1:
        cifar10_training_loader = []

        if noniid_ratio < 1.0:
            cifar10_training_subset_list = noniid_alllabel(cifar10_training, num_client, noniid_ratio, 100)

        if not collude_use_public:
            for i in range(num_client):
                if noniid_ratio == 1.0:
                    cifar10_training_subset = torch.utils.data.Subset(cifar10_training, list(range(i * (len(cifar10_training)//num_client), (i+1) * (len(cifar10_training)//num_client))))
                else:
                    cifar10_training_subset = DatasetSplit(cifar10_training, cifar10_training_subset_list[i])
                
                subset_training_loader = DataLoader(
                    cifar10_training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
                cifar10_training_loader.append(subset_training_loader)
        else:
            '''1 + collude + (n-2) vanilla clients, all training data is shared by n-1 clients'''
            # cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
            # subset_training_loader = DataLoader(
            #     cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            # cifar10_training_loader.append(subset_training_loader)
            # for i in range(num_client-1):
            #     cifar10_training_subset = torch.utils.data.Subset(cifar10_training, list(range(i * (len(cifar10_training)//(num_client-1)), (i+1) * (len(cifar10_training)//(num_client-1)))))
            #     subset_training_loader = DataLoader(
            #         cifar10_training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            #     cifar10_training_loader.append(subset_training_loader)
            # # switch the testloader to collude position
            # temp = cifar10_training_loader[0]
            # cifar10_training_loader[0] = cifar10_training_loader[1]
            # cifar10_training_loader[1] = temp

            '''1+ (n-1) * collude, the single client gets all training data'''
            subset_training_loader = DataLoader(
                cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            cifar10_training_loader.append(subset_training_loader)
            cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
            for i in range(num_client-1):
                subset_training_loader = DataLoader(
                    cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
                
                cifar10_training_loader.append(subset_training_loader)
    cifar10_training2 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    cifar10_training_mem = torch.utils.data.Subset(cifar10_training2, list(range(0, 5000)))
    xmem_training_loader = DataLoader(
        cifar10_training_mem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    cifar10_testing_mem = torch.utils.data.Subset(cifar10_training2, list(range(5000, 10000)))
    xmem_testing_loader = DataLoader(
        cifar10_testing_mem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


    return cifar10_training_loader, xmem_training_loader, xmem_testing_loader

def get_cifar10_testloader(batch_size=16, num_workers=2, shuffle=True, extra_cls_removed_dataset = False, cls_to_remove = 0):
    """ return training dataloader
    Args:
        mean: mean of cifar10 test dataset
        std: std of cifar10 test dataset
        path: path to cifar10 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar10_test_loader:torch dataloader object
    """
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
    ])

    transform_exlabel = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    cifar10_test2 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
    cifar10_training_nomem = torch.utils.data.Subset(cifar10_test2, list(range(0, len(cifar10_test2)//2)))
    nomem_training_loader = DataLoader(
        cifar10_training_nomem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    cifar10_testing_nomem = torch.utils.data.Subset(cifar10_test2, list(range(len(cifar10_test2)//2, len(cifar10_test2))))
    nomem_testing_loader = DataLoader(
        cifar10_testing_nomem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


    if extra_cls_removed_dataset:

        cifar10_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_exlabel)
        cifar10_cls_rm_loader, cifar10_cls_ex_loader = remove_class_loader(cifar10_training, cls_to_remove, batch_size, num_workers)
        return cifar10_test_loader, cifar10_cls_rm_loader, cifar10_cls_ex_loader
    return cifar10_test_loader, nomem_training_loader, nomem_testing_loader




def get_cifar100_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, collude_use_public = False, data_portion = 1.0, noniid_ratio = 1.0):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    
    indices = torch.randperm(len(cifar100_training))[:int(len(cifar100_training)* data_portion)]

    cifar100_training = torch.utils.data.Subset(cifar100_training, indices)
    
    if num_client == 1:
        cifar100_training_loader = [DataLoader(
            cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)]
    
    elif num_client > 1:
        cifar100_training_loader = []

        if noniid_ratio < 1.0:
            cifar100_training_subset_list = noniid_alllabel(cifar100_training, num_client, noniid_ratio, 100)
        

        if not collude_use_public:
            for i in range(num_client):
                if noniid_ratio == 1.0:
                    cifar100_training_subset = torch.utils.data.Subset(cifar100_training, list(range(i * (len(cifar100_training)//num_client), (i+1) * (len(cifar100_training)//num_client))))
                else:
                    cifar100_training_subset = DatasetSplit(cifar100_training, cifar100_training_subset_list[i])
                
                subset_training_loader = DataLoader(
                    cifar100_training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
                cifar100_training_loader.append(subset_training_loader)
        else:
            '''1 + collude + (n-2) vanilla clients, all training data is shared by n-1 clients'''
            # cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_train)
            # subset_training_loader = DataLoader(
            #     cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            # cifar100_training_loader.append(subset_training_loader)
            # for i in range(num_client-1):
            #     cifar100_training_subset = torch.utils.data.Subset(cifar100_training, list(range(i * (len(cifar100_training)//(num_client-1)), (i+1) * (len(cifar100_training)//(num_client-1)))))
            #     subset_training_loader = DataLoader(
            #         cifar100_training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            #     cifar100_training_loader.append(subset_training_loader)
            # # switch the testloader to collude position
            # temp = cifar100_training_loader[0]
            # cifar100_training_loader[0] = cifar100_training_loader[1]
            # cifar100_training_loader[1] = temp

            '''1+ (n-1) * collude, the single client gets all training data'''
            subset_training_loader = DataLoader(
                cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            cifar100_training_loader.append(subset_training_loader)
            cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_train)
            for i in range(num_client-1):
                subset_training_loader = DataLoader(
                    cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
                
                cifar100_training_loader.append(subset_training_loader)

    cifar100_training2 = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)

    cifar100_training_mem = torch.utils.data.Subset(cifar100_training2, list(range(0, 5000)))
    xmem_training_loader = DataLoader(
        cifar100_training_mem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    cifar100_testing_mem = torch.utils.data.Subset(cifar100_training2, list(range(5000, 10000)))
    xmem_testing_loader = DataLoader(
        cifar100_testing_mem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


    return cifar100_training_loader, xmem_training_loader, xmem_testing_loader



def get_cifar100_testloader(batch_size=16, num_workers=2, shuffle=True, extra_cls_removed_dataset = False, cls_to_remove = 0):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])

    transform_exlabel = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    cifar100_test2 = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_train)
    cifar100_training_nomem = torch.utils.data.Subset(cifar100_test2, list(range(0, len(cifar100_test2)//2)))
    nomem_training_loader = DataLoader(
        cifar100_training_nomem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    cifar100_testing_nomem = torch.utils.data.Subset(cifar100_test2, list(range(len(cifar100_test2)//2, len(cifar100_test2))))
    nomem_testing_loader = DataLoader(
        cifar100_testing_nomem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    if extra_cls_removed_dataset:
        cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_exlabel)
        cifar100_cls_rm_loader, cifar100_cls_ex_loader = remove_class_loader(cifar100_training, cls_to_remove, batch_size, num_workers)
        return cifar100_test_loader, cifar100_cls_rm_loader, cifar100_cls_ex_loader
    return cifar100_test_loader, nomem_training_loader, nomem_testing_loader




def get_SVHN_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, collude_use_public = False):
    """ return training dataloader
    Args:
        mean: mean of SVHN training dataset
        std: std of SVHN training dataset
        path: path to SVHN training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)
    ])
    #cifar00_training = SVHNTrain(path, transform=transform_train)
    SVHN_training = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)
    if num_client == 1:
        SVHN_training_loader = [DataLoader(
            SVHN_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)]
    elif num_client > 1:
        SVHN_training_loader = []
        if not collude_use_public:
            for i in range(num_client):
                SVHN_training_subset = torch.utils.data.Subset(SVHN_training, list(range(i * (len(SVHN_training)//num_client), (i+1) * (len(SVHN_training)//num_client))))
                subset_training_loader = DataLoader(
                    SVHN_training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
                SVHN_training_loader.append(subset_training_loader)
        else:
            '''1 + collude + (n-2) vanilla clients, all training data is shared by n-1 clients'''
            # SVHN_test = torchvision.datasets.SVHN(root='./data', train=False, download=True, transform=transform_train)
            # subset_training_loader = DataLoader(
            #     SVHN_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            # SVHN_training_loader.append(subset_training_loader)
            # for i in range(num_client-1):
            #     SVHN_training_subset = torch.utils.data.Subset(SVHN_training, list(range(i * (len(SVHN_training)//(num_client-1)), (i+1) * (len(SVHN_training)//(num_client-1)))))
            #     subset_training_loader = DataLoader(
            #         SVHN_training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            #     SVHN_training_loader.append(subset_training_loader)
            # # switch the testloader to collude position
            # temp = SVHN_training_loader[0]
            # SVHN_training_loader[0] = SVHN_training_loader[1]
            # SVHN_training_loader[1] = temp

            '''1+ (n-1) * collude, the single client gets all training data'''
            subset_training_loader = DataLoader(
                SVHN_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            SVHN_training_loader.append(subset_training_loader)
            SVHN_test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_train)
            for i in range(num_client-1):
                subset_training_loader = DataLoader(
                    SVHN_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
                
                SVHN_training_loader.append(subset_training_loader)
    SVHN_training2 = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_train)

    SVHN_training_mem = torch.utils.data.Subset(SVHN_training2, list(range(0, 5000)))
    xmem_training_loader = DataLoader(
        SVHN_training_mem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    SVHN_testing_mem = torch.utils.data.Subset(SVHN_training2, list(range(5000, 10000)))
    xmem_testing_loader = DataLoader(
        SVHN_testing_mem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


    return SVHN_training_loader, xmem_training_loader, xmem_testing_loader

def get_SVHN_testloader(batch_size=16, num_workers=2, shuffle=True, extra_cls_removed_dataset = False, cls_to_remove = 0):
    """ return training dataloader
    Args:
        mean: mean of SVHN test dataset
        std: std of SVHN test dataset
        path: path to SVHN test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: SVHN_test_loader:torch dataloader object
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)
    ])

    transform_exlabel = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    SVHN_test = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
    print(len(SVHN_test))
    SVHN_test_loader = DataLoader(
        SVHN_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    SVHN_test2 = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_train)
    SVHN_training_nomem = torch.utils.data.Subset(SVHN_test2, list(range(0, len(SVHN_test2)//2)))
    nomem_training_loader = DataLoader(
        SVHN_training_nomem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    SVHN_testing_nomem = torch.utils.data.Subset(SVHN_test2, list(range(len(SVHN_test2)//2, len(SVHN_test2))))
    nomem_testing_loader = DataLoader(
        SVHN_testing_nomem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


    if extra_cls_removed_dataset:

        SVHN_training = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=transform_exlabel)
        SVHN_cls_rm_loader, SVHN_cls_ex_loader = remove_class_loader(SVHN_training, cls_to_remove, batch_size, num_workers)
        return SVHN_test_loader, SVHN_cls_rm_loader, SVHN_cls_ex_loader
    return SVHN_test_loader, nomem_training_loader, nomem_testing_loader








################


def get_celeba_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, collude_use_public = False):
    """ return training dataloader
    Args:
        mean: mean of celeba training dataset
        std: std of celeba training dataset
        path: path to celeba training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    celeba_training = torchvision.datasets.CelebA(root='./data', train=True, download=True, transform=transform_train)
    if num_client == 1:
        celeba_training_loader = [DataLoader(
            celeba_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)]
    
    elif num_client > 1:
        celeba_training_loader = []
        if not collude_use_public:
            for i in range(num_client):
                celeba_training_subset = torch.utils.data.Subset(celeba_training, list(range(i * (len(celeba_training)//num_client), (i+1) * (len(celeba_training)//num_client))))
                subset_training_loader = DataLoader(
                    celeba_training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
                celeba_training_loader.append(subset_training_loader)
        else:
            '''1 + collude + (n-2) vanilla clients, all training data is shared by n-1 clients'''
            # celeba_test = torchvision.datasets.CelebA(root='./data', train=False, download=True, transform=transform_train)
            # subset_training_loader = DataLoader(
            #     celeba_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            # celeba_training_loader.append(subset_training_loader)
            # for i in range(num_client-1):
            #     celeba_training_subset = torch.utils.data.Subset(celeba_training, list(range(i * (len(celeba_training)//(num_client-1)), (i+1) * (len(celeba_training)//(num_client-1)))))
            #     subset_training_loader = DataLoader(
            #         celeba_training_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            #     celeba_training_loader.append(subset_training_loader)
            # # switch the testloader to collude position
            # temp = celeba_training_loader[0]
            # celeba_training_loader[0] = celeba_training_loader[1]
            # celeba_training_loader[1] = temp

            '''1+ (n-1) * collude, the single client gets all training data'''
            subset_training_loader = DataLoader(
                celeba_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
            celeba_training_loader.append(subset_training_loader)
            celeba_test = torchvision.datasets.CelebA(root='./data', train=False, download=True, transform=transform_train)
            for i in range(num_client-1):
                subset_training_loader = DataLoader(
                    celeba_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
                
                celeba_training_loader.append(subset_training_loader)

    celeba_training2 = torchvision.datasets.CelebA(root='./data', train=True, download=True, transform=transform_train)

    celeba_training_mem = torch.utils.data.Subset(celeba_training2, list(range(0, 5000)))
    xmem_training_loader = DataLoader(
        celeba_training_mem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    celeba_testing_mem = torch.utils.data.Subset(celeba_training2, list(range(5000, 10000)))
    xmem_testing_loader = DataLoader(
        celeba_testing_mem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


    return celeba_training_loader, xmem_training_loader, xmem_testing_loader

def get_celeba_testloader(batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of celeba test dataset
        std: std of celeba test dataset
        path: path to celeba test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: celeba_test_loader:torch dataloader object
    """
    transform_train = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    #celeba_test = CIFAR100Test(path, transform=transform_test)
    celeba_test = torchvision.datasets.CelebA(root='./data', train=False, download=True, transform=transform_test)
    celeba_test_loader = DataLoader(
        celeba_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    celeba_test2 = torchvision.datasets.CelebA(root='./data', train=False, download=True, transform=transform_train)
    celeba_training_nomem = torch.utils.data.Subset(celeba_test2, list(range(0, len(celeba_test2)//2)))
    nomem_training_loader = DataLoader(
        celeba_training_nomem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    celeba_testing_nomem = torch.utils.data.Subset(celeba_test2, list(range(len(celeba_test2)//2, len(celeba_test2))))
    nomem_testing_loader = DataLoader(
        celeba_testing_nomem, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)


    return celeba_test_loader, nomem_training_loader, nomem_testing_loader
