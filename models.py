from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv


class MLP(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0.5,
                 activation="relu"):
        super(MLP, self).__init__()
        self.fc1 = Linear(num_features, hidden_size)
        self.fc2 = Linear(hidden_size, num_classes)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class GCN(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0.5,
                 activation="relu"):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0.5,
                 activation="relu",
                 num_heads=8):
        super(GAT, self).__init__()
        self.conv1 = GATConv(
            num_features, hidden_size, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(
            hidden_size * num_heads, num_classes, dropout=dropout)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def _one_hot(idx, num_class):
    return torch.zeros(len(idx), num_class).to(idx.device).scatter_(
        1, idx.unsqueeze(1), 1.)


class LSM(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 hidden_x,
                 dropout=0.5,
                 activation="relu",
                 neg_ratio=1.0):
        super(LSM, self).__init__()
        self.p_y_x = MLP(num_features, num_classes, hidden_size, dropout,
                         activation)
        self.x_enc = Linear(num_features, hidden_x)
        self.p_e_xy = Linear(2 * (hidden_x + num_classes), 1)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.neg_ratio = neg_ratio

    def forward(self, data):
        y_log_prob = self.p_y_x(data)
        y_prob = torch.exp(y_log_prob)
        y_prob = torch.where(
            data.train_mask.unsqueeze(1), _one_hot(data.y, y_prob.size(1)),
            y_prob)
        x = F.dropout(data.x, p=self.dropout, training=self.training)
        x = self.activation(self.x_enc(x))

        # Positive edges.
        y_query = F.embedding(data.edge_index[0], y_prob)
        y_key = F.embedding(data.edge_index[1], y_prob)
        x_query = F.embedding(data.edge_index[0], x)
        x_key = F.embedding(data.edge_index[1], x)
        xy = torch.cat([x_query, x_key, y_query, y_key], dim=1)
        e_pred_pos = self.p_e_xy(xy)

        # Negative edges.
        e_pred_neg = None
        if self.neg_ratio > 0:
            num_edges_pos = data.edge_index.size(1)
            num_nodes = data.x.size(0)
            num_edges_neg = int(self.neg_ratio * num_edges_pos)
            edge_index_neg = torch.randint(num_nodes,
                                           (2, num_edges_neg)).to(x.device)
            y_query = F.embedding(edge_index_neg[0], y_prob)
            y_key = F.embedding(edge_index_neg[1], y_prob)
            x_query = F.embedding(edge_index_neg[0], x)
            x_key = F.embedding(edge_index_neg[1], x)
            xy = torch.cat([x_query, x_key, y_query, y_key], dim=1)
            e_pred_neg = self.p_e_xy(xy)

        return e_pred_pos, e_pred_neg, y_log_prob

    def nll_generative(self, data, post_y_log_prob):
        e_pred_pos, e_pred_neg, y_log_prob = self.forward(data)
        # unlabel_mask = data.val_mask + data.test_mask
        unlabel_mask = torch.ones_like(data.train_mask) - data.train_mask

        # nll of p_g_xy
        nll_p_g_xy = -torch.mean(F.logsigmoid(e_pred_pos))
        if e_pred_neg is not None:
            nll_p_g_xy += -torch.mean(F.logsigmoid(-e_pred_neg))

        # nll of p_y_x
        nll_p_y_x = F.nll_loss(y_log_prob[data.train_mask],
                               data.y[data.train_mask])
        nll_p_y_x += -torch.mean(
            torch.exp(post_y_log_prob[unlabel_mask]) *
            y_log_prob[unlabel_mask])

        # nll of q_y_xg
        nll_q_y_xg = -torch.mean(
            torch.exp(post_y_log_prob[unlabel_mask]) *
            post_y_log_prob[unlabel_mask])

        return nll_p_g_xy + nll_p_y_x + nll_q_y_xg


class SBM(torch.nn.Module):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0.5,
                 activation="relu",
                 p0=0.9,
                 p1=0.1,
                 neg_ratio=1.0):
        super(SBM, self).__init__()
        self.p_y_x = MLP(num_features, num_classes, hidden_size, dropout,
                         activation)
        self.p0 = p0
        self.p1 = p1
        self.neg_ratio = neg_ratio

    def forward(self, data):
        y_log_prob = self.p_y_x(data)
        y_prob = torch.exp(y_log_prob)
        y_prob = torch.where(
            data.train_mask.unsqueeze(1), _one_hot(data.y, y_prob.size(1)),
            y_prob)

        # Positive edges.
        y_query_pos = F.embedding(data.edge_index[0], y_prob)
        y_key_pos = F.embedding(data.edge_index[1], y_prob)

        # Negative edges.
        y_query_neg = None
        y_key_neg = None
        if self.neg_ratio > 0:
            num_edges_pos = data.edge_index.size(1)
            num_nodes = data.x.size(0)
            num_edges_neg = int(self.neg_ratio * num_edges_pos)
            edge_index_neg = torch.randint(num_nodes, (2, num_edges_neg)).to(
                y_prob.device)
            y_query_neg = F.embedding(edge_index_neg[0], y_prob)
            y_key_neg = F.embedding(edge_index_neg[1], y_prob)

        return y_query_pos, y_key_pos, y_query_neg, y_key_neg, y_log_prob

    def nll_generative(self, data, post_y_log_prob):
        (y_query_pos, y_key_pos, y_query_neg, y_key_neg,
         y_log_prob) = self.forward(data)
        # unlabel_mask = data.val_mask + data.test_mask
        unlabel_mask = torch.ones_like(data.train_mask) - data.train_mask

        # nll of p_g_y
        nll_p_g_y = -torch.mean(y_query_pos * y_key_pos) * np.log(
            self.p0 / self.p1)
        if y_query_neg is not None:
            nll_p_g_y += -torch.mean(y_query_neg * y_key_neg) * np.log(
                (1 - self.p0) / (1 - self.p1))

        # nll of p_y_x
        nll_p_y_x = F.nll_loss(y_log_prob[data.train_mask],
                               data.y[data.train_mask])
        nll_p_y_x += -torch.mean(
            torch.exp(post_y_log_prob[unlabel_mask]) *
            y_log_prob[unlabel_mask])

        # nll of q_y_xg
        nll_q_y_xg = -torch.mean(
            torch.exp(post_y_log_prob[unlabel_mask]) *
            post_y_log_prob[unlabel_mask])

        return nll_p_g_y + nll_p_y_x + nll_q_y_xg


class GenGNN(torch.nn.Module):
    def __init__(self, gen_config, post_config):
        super(GenGNN, self).__init__()
        self.gen_type = gen_config.pop("type")
        if self.gen_type == "lsm":
            self.gen = LSM(**gen_config)
        elif self.gen_type == "sbm":
            self.gen = SBM(**gen_config)
        else:
            raise NotImplementedError(
                "Generative model type %s not supported." % self.gen_type)

        self.post_type = post_config.pop("type")
        if self.post_type == "gcn":
            self.post = GCN(**post_config)
        elif self.post_type == "gat":
            self.post = GAT(**post_config)
        else:
            raise NotImplementedError(
                "Generative model type %s not supported." % self.post_type)

    def forward(self, data):
        return self.post(data)
