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

FashionMNIST_mean = (0.5,)#(0.2860,)  # equals np.mean(train_set.train_data)
FashionMNIST_std = (0.5,)#(0.3530,)  # equals np.std(train_set.train_data)

def normalise(x, mean=FashionMNIST_mean, std=FashionMNIST_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean
    x /= std
    return x

def transpose(x, source='NHWC', target='NCHW'):
    '''
    N: batch size
    H: height
    W: weight
    C: channel
    '''
    return x.transpose([source.index(d) for d in target])

def _3D_to_4(x):
    '''
    :param x: For mnist, it is a tensor of shape (len, 28, 28)
    :return: a tensor of shape (len, 1, 28, 28)
    '''
    return x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])


class FashionMNIST_labeled(torchvision.datasets.FashionMNIST):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(FashionMNIST_labeled, self).__init__(root, train=train,
                                                   transform=transform, target_transform=target_transform,
                                                   download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = _3D_to_4(normalise(self.data))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class FashionMNIST_unlabeled(FashionMNIST_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True):
        super(FashionMNIST_unlabeled, self).__init__(root, indexs, train=train,
                                                     transform=transform, target_transform=target_transform,
                                                     download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])




def get_fashionMNIST_data(num_labeled, positive_label_list, root, transform_train=None, transform_val=None):
    base_dataset = torchvision.datasets.FashionMNIST(root, train=True, download=True)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, num_labeled,
                                                                         positive_label_list)
    target_transform = lambda x: 1 if x in positive_label_list else 0
    train_labeled_dataset = FashionMNIST_labeled(root, train_labeled_idxs, train=True, transform=transform_train,
                                            target_transform=target_transform)
    train_unlabeled_dataset = FashionMNIST_unlabeled(root, train_unlabeled_idxs, train=True, transform=transform_train,
                                                target_transform=target_transform)
    val_unlabeled_dataset = FashionMNIST_unlabeled(root, val_idxs, train=True, transform=transform_val, download=True,
                                              target_transform=target_transform)
    val_labeled_idxs = get_val_labeled(base_dataset.targets, val_idxs, num_labeled*5000/55000, positive_label_list)
    val_labeled_dataset = FashionMNIST_labeled(root, val_labeled_idxs, train=True, transform=transform_val, download=True,
                                          target_transform=target_transform)
    test_dataset = FashionMNIST_labeled(root, train=False, transform=transform_val, download=True,
                                   target_transform=target_transform)

    idx = (train_labeled_idxs, train_unlabeled_idxs, val_labeled_idxs, val_idxs)
    return train_labeled_dataset, train_unlabeled_dataset, val_labeled_dataset, val_unlabeled_dataset, test_dataset, idx


def get_fashionMNIST_loaders(positive_label_list, batch_size=500, num_labeled=3000):
    train_labeled_dataset, train_unlabeled_dataset, val_labeled_dataset, val_unlabeled_dataset, test_dataset, idx = get_fashionMNIST_data(
        num_labeled=num_labeled,
        positive_label_list=positive_label_list,
        root=os.path.join(os.getcwd(), 'data'))
    p_loader = DataLoader(dataset=train_labeled_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    x_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    val_p_loader = DataLoader(dataset=val_labeled_dataset, batch_size=batch_size, shuffle=False)
    val_x_loader = DataLoader(dataset=val_unlabeled_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return x_loader, p_loader, val_x_loader, val_p_loader,  test_loader, idx













