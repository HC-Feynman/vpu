import os
import numpy as np
import torchvision
from torch.utils.data import DataLoader

stl10_mean = (0.5, 0.5, 0.5)
stl10_std = (0.5, 0.5, 0.5)


def normalise(x, mean=stl10_mean, std=stl10_std):
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


class STL10_labeled(torchvision.datasets.STL10):

    def __init__(self, root, indexs=None, split='train+unlabeled', transform=None, target_transform=None,
                 download=False):
        super(STL10_labeled, self).__init__(root, split=split,
                                            transform=transform, target_transform=target_transform,
                                            download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
        self.data = transpose(self.data, source='NCHW', target='NHWC')
        self.data = transpose(normalise(self.data))

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class STL10_unlabeled(STL10_labeled):

    def __init__(self, root, indexs, split='train+unlabeled',
                 transform=None, target_transform=None,
                 download=False):
        super(STL10_unlabeled, self).__init__(root, indexs, split=split,
                                              transform=transform, target_transform=target_transform,
                                              download=download)
        self.labels = np.array([-1 for i in range(len(self.labels))])


def labeled_split_train_val(labels, num_labeled, positive_label_list):
    train_labeled_idxs = []
    val_labeled_idxs = []
    train_unlabeled_idxs = []
    val_unlabeled_idxs = []
    n_labeled_per_class = int(num_labeled / len(positive_label_list))

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        if i in positive_label_list:
            train_labeled_idxs.extend(idxs[:n_labeled_per_class])
            val_labeled_idxs.extend(idxs[n_labeled_per_class:])
        else:
            train_unlabeled_idxs.extend(idxs[:n_labeled_per_class])
            val_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
    idxs = np.where(labels == -1)[0]
    np.random.shuffle(idxs)
    num_unlabeled_val = int(100000 * len(val_labeled_idxs) / (500 * len(positive_label_list)))
    val_unlabeled_idxs.extend(idxs[:num_unlabeled_val])
    train_unlabeled_idxs.extend(idxs[num_unlabeled_val:])
    train_unlabeled_idxs.extend(train_labeled_idxs)
    val_unlabeled_idxs.extend(val_labeled_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_labeled_idxs, val_unlabeled_idxs


def get_stl10_data(num_labeled, positive_label_list, root, transform_train=None, transform_val=None):
    base_dataset = torchvision.datasets.STL10(root, split='train+unlabeled', download=True)
    train_labeled_idxs, train_unlabeled_idxs, val_labeled_idxs, val_unlabeled_idxs = labeled_split_train_val(
        base_dataset.labels, num_labeled, positive_label_list)
    target_transform = lambda x: 1 if x in positive_label_list else 0
    train_labeled_dataset = STL10_labeled(root, train_labeled_idxs, split='train+unlabeled', transform=transform_train,
                                          target_transform=target_transform)
    train_unlabeled_dataset = STL10_unlabeled(root, train_unlabeled_idxs, split='train+unlabeled',
                                              transform=transform_train,
                                              target_transform=target_transform)
    val_unlabeled_dataset = STL10_unlabeled(root, val_unlabeled_idxs, split='train+unlabeled', transform=transform_val,
                                            download=True,
                                            target_transform=target_transform)
    val_labeled_dataset = STL10_labeled(root, val_labeled_idxs, split='train+unlabeled', transform=transform_val,
                                        download=True,
                                        target_transform=target_transform)
    test_dataset = STL10_labeled(root, split='test', transform=transform_val, download=True,
                                 target_transform=target_transform)

    idx = (train_labeled_idxs, train_unlabeled_idxs, val_labeled_idxs, val_unlabeled_idxs)
    return train_labeled_dataset, train_unlabeled_dataset, val_labeled_dataset, val_unlabeled_dataset, test_dataset, idx


def get_stl10_loaders(positive_label_list, batch_size=500, num_labeled=2250):
    train_labeled_dataset, train_unlabeled_dataset, val_labeled_dataset, val_unlabeled_dataset, test_dataset, idx = get_stl10_data(
        num_labeled=num_labeled,
        positive_label_list=positive_label_list,
        root=os.path.join(os.getcwd(), 'data'))
    p_loader = DataLoader(dataset=train_labeled_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    x_loader = DataLoader(dataset=train_unlabeled_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
    val_p_loader = DataLoader(dataset=val_labeled_dataset, batch_size=batch_size, shuffle=False)
    val_x_loader = DataLoader(dataset=val_unlabeled_dataset, batch_size=batch_size, shuffle=False)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return x_loader, p_loader, val_x_loader, val_p_loader, test_loader, idx
