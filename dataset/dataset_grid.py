import os
import random
import pandas as pd
import numpy as np
import torch.autograd
from sklearn import preprocessing
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split



def get_grid_loaders(batch_size=500, num_labeled=1000, positive_label_list=[1], test_proportion=0.4,if_normalized=False):
    val_x_size = 1000
    data_frame = pd.read_csv(os.getcwd() + '/data/stable_.csv', sep=',', header=None)
    data_ndarray = data_frame.values
    if if_normalized:
        min_max_scaler = preprocessing.MinMaxScaler()
        data_ndarray = np.hstack((min_max_scaler.fit_transform(data_ndarray[:,0:-1]),data_ndarray[:,-1].reshape(-1,1)))
    train_ndarray, test_ndarray = train_test_split(data_ndarray, test_size=test_proportion)  # random_state=0 in DAN
    val_p_size = num_labeled / len(train_ndarray) * val_x_size

    # the set of all training data, whose labels are unprocessed
    train_set_unprocessed = []
    for data in train_ndarray:
        x_tr = torch.FloatTensor(data[:-1].astype(np.float64))
        y_tr = data[-1]
        train_set_unprocessed.append((x_tr, y_tr))


    # sets of training data with processed labels
    random.shuffle(train_set_unprocessed)
    train_p_set = []  # with label
    train_x_set = []  # with label
    val_p_set = []
    val_x_set = []
    cnt_in_p = 0
    cnt_val_p = 0
    cnt_val_x = 0
    for data, target in train_set_unprocessed:
        if cnt_val_x < val_x_size:
            cnt_val_x += 1
            if target in positive_label_list:
                target = 1
                val_x_set.append((data, target))
                if cnt_val_p < val_p_size:
                    val_p_set.append((data, target))
                    cnt_val_p += 1
            else:
                target = 0
                val_x_set.append((data, target))
        else:
            if target in positive_label_list:
                target = 1
                train_x_set.append((data, target))
                if cnt_in_p < num_labeled:
                    train_p_set.append((data, target))
                    cnt_in_p += 1
            else:
                target = 0
                train_x_set.append((data, target))

    p_loader = DataLoader(dataset=train_p_set, batch_size=batch_size, shuffle=True,drop_last=True)
    x_loader = DataLoader(dataset=train_x_set, batch_size=batch_size, shuffle=True,drop_last=True)
    val_p_loader = DataLoader(dataset=val_p_set, batch_size=batch_size, shuffle=True)
    val_x_loader = DataLoader(dataset=val_x_set, batch_size=batch_size, shuffle=True)

    test_set_unprocessed = []
    for data in test_ndarray:
        x_te = torch.FloatTensor(data[:-1].astype(np.float64))
        y_te = data[-1]
        test_set_unprocessed.append((x_te, y_te))
    test_set = []
    for data, target in test_set_unprocessed:
        if target in positive_label_list:
            target = 1
            test_set.append((data, target))
        else:
            target = 0
            test_set.append((data, target))
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    return x_loader, p_loader, val_x_loader, val_p_loader, test_loader

