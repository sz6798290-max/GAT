# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):

    def __init__(self, in_size, out_size, weight_dropout=0):
        super().__init__()
        self.weight = nn.Linear(in_size, out_size, bias=False)
        self.weight_dropout = weight_dropout

    def forward(self, x, adj):
        m = self.weight(x)
        if self.weight_dropout > 0:
            m = F.dropout(m, self.weight_dropout, training=self.training)
            adj = F.dropout(adj, self.weight_dropout, training=self.training)
        m = torch.mm(adj, m)
        return m


class AttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, label_features, label_out_features, dropout,
                 weight_dropout=0):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.label_features = label_features
        self.label_features = label_out_features
        self.weight_dropout = weight_dropout

        self.transformation = nn.Linear(in_features, out_features, bias=False)
        self.W = GraphConvolution(label_features, label_out_features)

        self.h_1 = nn.Linear(out_features, 1, bias=False)
        self.h_2 = nn.Linear(out_features, 1, bias=False)

        self.trans = nn.Linear(label_out_features, out_features, bias=False)

    def forward(self, x, adj, y, gcn_adj):
        y = self.W(y, gcn_adj)
        x = self.transformation(x)

        h_1 = self.h_1(self.trans(y))
        h_2 = self.h_2(x)
        coefs = F.softmax(torch.tanh(torch.transpose(h_2, 0, 1) + h_1) + adj, dim=1)

        z = x
        z = F.dropout(z, self.weight_dropout, training=self.training)
        coefs = F.dropout(coefs, self.weight_dropout, training=self.training)
        z = torch.mm(coefs, z)

        return z, y


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, label_features, label_out_features, dropout,
                 weight_dropout=0):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.label_features = label_features
        self.label_features = label_out_features
        self.weight_dropout = weight_dropout

        self.transformation = nn.Linear(in_features, out_features, bias=False)
        self.W = GraphConvolution(label_features, label_out_features)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data)

        self.trans = nn.Linear(label_out_features, out_features, bias=False)

        self.special_spmm = SpecialSpmm()

    def forward(self, x, adj, y, gcn_adj):
        N = x.size()[0]
        edge = adj.nonzero().t()

        y = self.W(y, gcn_adj)
        assert not torch.isnan(y).any()

        x = self.transformation(x)
        # N x out
        assert not torch.isnan(x).any()

        h = torch.cat((self.trans(y)[edge[0, :], :], x[edge[1, :], :]), dim=1).t()
        # 2*D x E

        t = torch.mm(self.a, h).squeeze()
        h = torch.exp(torch.tanh(t))
        assert not torch.isnan(h).any()
        # E

        rowsum = self.special_spmm(edge, h, torch.Size([N, N]), torch.ones(size=(N, 1), device='cuda'))
        # N x 1

        h = F.dropout(h, self.dropout, training=self.training)
        # E
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.special_spmm(edge, h, torch.Size([N, N]), x)
        assert not torch.isnan(x).any()
        # N x out

        x = x.div(rowsum)
        assert not torch.isnan(x).any()

        return x, y
