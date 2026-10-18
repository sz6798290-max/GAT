# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, AttentionLayer, SpAttentionLayer


class SLGAT(nn.Module):
    def __init__(self, num_feature, hidden_dim, num_class, class_hidden,
                 adj, gcn_adj, input_dropout, dropout, weight_dropout=0):
        super().__init__()
        self.num_feature = num_feature
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.adj = adj
        self.gcn_adj = gcn_adj

        self.m1 = AttentionLayer(num_feature, hidden_dim, num_class, class_hidden, input_dropout,
                                 weight_dropout=weight_dropout)
        self.m2 = AttentionLayer(hidden_dim, num_class, class_hidden, class_hidden, input_dropout,
                                 weight_dropout=weight_dropout)

        self.g1 = GraphConvolution(num_feature, hidden_dim, weight_dropout=weight_dropout)
        self.g2 = GraphConvolution(hidden_dim, num_class, weight_dropout=weight_dropout)

        self.input_dropout = input_dropout
        self.dropout = dropout

    def forward_pre_train(self, x):
        x = F.dropout(x, self.input_dropout, training=self.training)
        x = self.g1(x, self.gcn_adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.g2(x, self.gcn_adj)
        return x

    def forward(self, x):
        z = self.forward_pre_train(x)
        preds = torch.softmax(z, dim=-1)
        x, y = self.forward_with_labels(x, preds)
        return (z + x) * 0.5

    def forward_with_labels(self, x, y):
        x = F.dropout(x, self.input_dropout, training=self.training)
        y = F.dropout(y, self.input_dropout, training=self.training)
        x, y = self.m1(x, self.adj, y, self.gcn_adj)
        x = F.relu(x)
        y = F.relu(y)
        x = F.dropout(x, self.dropout, training=self.training)
        y = F.dropout(y, self.dropout, training=self.training)
        x, y = self.m2(x, self.adj, y, self.gcn_adj)
        return x, y


class SpSLGAT(nn.Module):
    def __init__(self, num_feature, hidden_dim, num_class, class_hidden,
                 adj, gcn_adj, input_dropout, dropout, weight_dropout=0):
        super().__init__()
        self.num_feature = num_feature
        self.hidden_dim = hidden_dim
        self.num_class = num_class
        self.adj = adj
        self.gcn_adj = gcn_adj

        self.m1 = SpAttentionLayer(num_feature, hidden_dim, num_class, class_hidden, input_dropout,
                                   weight_dropout=weight_dropout)
        self.m2 = SpAttentionLayer(hidden_dim, num_class, class_hidden, class_hidden, input_dropout,
                                   weight_dropout=weight_dropout)

        self.g1 = GraphConvolution(num_feature, hidden_dim)
        self.g2 = GraphConvolution(hidden_dim, num_class)

        self.input_dropout = input_dropout
        self.dropout = dropout

    def forward_pre_train(self, x):
        x = F.dropout(x, self.input_dropout, training=self.training)
        x = self.g1(x, self.gcn_adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.g2(x, self.gcn_adj)
        return x

    def forward(self, x):
        z = self.forward_pre_train(x)
        preds = torch.softmax(z, dim=-1)
        x, y = self.forward_with_labels(x, preds)
        return (z + x) * 0.5

    def forward_with_labels(self, x, y):
        x = F.dropout(x, self.input_dropout, training=self.training)
        y = F.dropout(y, self.input_dropout, training=self.training)
        x, y = self.m1(x, self.adj, y, self.gcn_adj)
        x = F.relu(x)
        y = F.relu(y)
        x = F.dropout(x, self.dropout, training=self.training)
        y = F.dropout(y, self.dropout, training=self.training)
        x, y = self.m2(x, self.adj, y, self.gcn_adj)
        return x, y
