# coding=utf-8

import torch


def adj_row_normalize(adj):
    o = torch.mm((torch.sum(adj, dim=1) ** (-1)).unsqueeze(1),
                 torch.ones(1, adj.size()[0]).type_as(adj))
    return adj * o


def adj_bias_normalize(adj):
    neg_zeros = -9e15 * torch.ones(adj.size()).type_as(adj)
    zeros = torch.zeros(adj.size()).type_as(adj)
    adj = torch.where(adj == 0, neg_zeros, adj)
    adj = torch.where(adj > 0, zeros, adj)
    return adj
