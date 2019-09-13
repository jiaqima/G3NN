import os

import numpy as np
import torch
import scipy.sparse as sps

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


def load_data(dataset="cora",
              num_labels_per_class=20,
              missing_edge=False,
              verbose=0):
    # Load data.
    path = os.path.join("data", dataset)
    if verbose:
        print("loading data from %s. %d labels per class." %
              (path, num_labels_per_class))
    assert dataset in ["cora", "pubmed", "citeseer"]
    dataset = Planetoid(
        root=path, name=dataset, transform=T.NormalizeFeatures())

    data = dataset[0]
    data.num_classes = dataset.num_classes

    if missing_edge:
        assert num_labels_per_class == 20
        test_idx = data.test_mask.nonzero().squeeze().numpy()
        edge_index = data.edge_index.numpy()
        num_nodes = data.y.size(0)
        adj = sps.csc_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
        adj_mask = np.ones(num_nodes)
        adj_mask[test_idx] = 0
        adj_mask = sps.diags(adj_mask, format="csr")
        adj = adj_mask.dot(adj).dot(adj_mask.tocsc()).tocoo()
        edge_index = np.concatenate([adj.row.reshape(1, -1), adj.col.reshape(1, -1)], axis=0)
        data.edge_index = torch.LongTensor(edge_index)

    # Original Planetoid setting.
    if num_labels_per_class == 20:
        return data

    # Get one-hot labels.
    temp = data.y.numpy()
    labels = np.zeros((len(temp), temp.max() + 1))
    for i in range(len(labels)):
        labels[i, temp[i]] = 1

    all_idx = list(range(len(labels)))

    # Select a fixed number of training data per class.
    idx_train = []
    class_cnt = np.zeros(
        labels.shape[1])  # number of nodes selected for each class
    for i in all_idx:
        if (class_cnt >= num_labels_per_class).all():
            break
        if ((class_cnt + labels[i]) > num_labels_per_class).any():
            continue
        class_cnt += labels[i]
        idx_train.append(i)
    if verbose:
        print("number of training data: ", len(idx_train))

    train_mask = np.zeros((len(labels), ), dtype=int)
    val_mask = np.zeros((len(labels), ), dtype=int)
    test_mask = np.zeros((len(labels), ), dtype=int)
    for i in all_idx:
        if i in idx_train:
            train_mask[i] = 1
        elif sum(val_mask) < 500:  # select 500 validation data
            val_mask[i] = 1
        else:
            test_mask[i] = 1
    data.train_mask = torch.ByteTensor(train_mask)
    data.val_mask = torch.ByteTensor(val_mask)
    data.test_mask = torch.ByteTensor(test_mask)

    return data
