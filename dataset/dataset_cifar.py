"""
Partly imported from

https://github.com/YU1ut/MixMatch-pytorch/blob/master/dataset/cifar10.py
"""

import os
import numpy as np
import torchvision
from torch.utils.data import DataLoader


def get_val_labeled(labels, val_idxs, n_val_labeled, positive_label_list):
    val_labeled_idxs = []
    np.random.shuffle(val_idxs)
    labels = np.array(labels)
    n_labeled_per_class = int(n_val_labeled / len(positive_label_list))
    for i in positive_label_list:
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        val_labeled_idxs.extend(idxs[0:n_labeled_per_class])
    return val_labeled_idxs


def train_val_split(labels, n_labeled, positive_label_list):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []
    n_labeled_per_class = int(n_labeled / len(positive_label_list))

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        if i in positive_label_list:
            train_labeled_idxs.extend(idxs[:n_labeled_per_class])
            train_unlabeled_idxs.extend(idxs[0:-500])
        else:
            train_unlabeled_idxs.extend(idxs[0:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


cifar10_mean = (0.5, 0.5, 0.5)
cifar10_std = (0.5, 0.5, 0.5)


def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean * 255
    x *= 1.0 / (255 * std)
    return x


def transpose(x, source='NHWC', target='NCHW'):
    '''
    N: batch size
    H: height
    W: weight
    C: channel
    '''
    return x.transpose([source.index(d) for d in target])


class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalise(self.data))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                                                transform=transform, target_transform=target_transform,
                                                download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])


def get_cifar10_data(num_labeled, positive_label_list, root, transform_train=None, transform_val=None):
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=True)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, num_labeled,
                                                                         positive_label_list)
    target_transform = lambda x: 1 if x in positive_label_list else 0
    train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train,
                                            target_transform=target_transform)
    train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True, transform=transform_train,
                                                target_transform=target_transform)
    val_unlabeled_dataset = CIFAR10_unlabeled(root, val_idxs, train=True, transform=transform_val, download=True,
                                              target_transform=target_transform)
    val_labeled_idxs = get_val_labeled(base_dataset.targets, val_idxs, num_labeled * 5000 / 45000, positive_label_list)
    val_labeled_dataset = CIFAR10_labeled(root, val_labeled_idxs, train=True, transform=transform_val, download=True,
                                          target_transform=target_transform)
    test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=True,
                                   target_transform=target_transform)

    idx = (train_labeled_idxs, train_unlabeled_idxs, val_labeled_idxs, val_idxs)
    return train_labeled_dataset, train_unlabeled_dataset, val_labeled_dataset, val_unlabeled_dataset, test_dataset, idx


def get_cifar10_loaders(positive_label_list, batch_size=500, num_labeled=3000):
    train_labeled_dataset, train_unlabeled_dataset, val_labeled_dataset, val_unlabeled_dataset, test_dataset, idx = get_cifar10_data(
        num_labeled=num_labeled,
        positive_label_list=positive_label_list,
        root=os.path.join(os.getcwd(), 'data'))
    p_loader = DataLoader(dataset=train_labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    x_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_p_loader = DataLoader(dataset=val_labeled_dataset, batch_size=batch_size, shuffle=False)
    val_x_loader = DataLoader(dataset=val_unlabeled_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return x_loader, p_loader, val_x_loader, val_p_loader, test_loader, idx
