# coding=utf-8

from os.path import join, curdir

import math
import numpy as np
import scipy.sparse as sp
import torch

import pickle


def load_gcn_graph(dataset='cora'):
    adj_norm = True
    path = join(curdir, 'data', dataset)
    with open(join(path, 'net.pkl'), 'rb') as f:
        adj = pickle.load(f)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    N = adj.shape[0]
    us, vs, ws = [], [], []
    d = {}
    for i in range(0, N):
        for j in range(adj.indptr[i], adj.indptr[i + 1]):
            d[i] = d.get(i, 0) + 1
            us.append(i)
            vs.append(adj.indices[j])
            w = adj.data[j]
            if w >= 1:
                w = 1
            elif w < 0:
                print(w, 'error')
            ws.append(w)
    if adj_norm:
        for i in range(0, len(us)):
            ws[i] = ws[i] / (math.sqrt(d[us[i]] * d[vs[i]]))
    index = torch.LongTensor([us, vs])
    value = torch.FloatTensor(ws)
    adj = torch.sparse.FloatTensor(index, value, torch.Size([N, N]))

    return adj


def load_gnn_data(dataset='cora', sparse=False):
    path = join(curdir, 'data', dataset)
    print('Loading {} dataset...'.format(dataset))
    if dataset == 'pubmed':
        adj_norm = True
    else:
        adj_norm = False

    with open(join(path, 'net.pkl'), 'rb') as f:
        adj = pickle.load(f)
    with open(join(path, 'label.pkl'), 'rb') as f:
        labels = pickle.load(f)
    labels_onehot = labels_to_onehot(labels)
    with open(join(path, 'feature.pkl'), 'rb') as f:
        features = pickle.load(f)
    with open(join(path, 'train.pkl'), 'rb') as f:
        idx_train = pickle.load(f)
    with open(join(path, 'dev.pkl'), 'rb') as f:
        idx_val = pickle.load(f)
    with open(join(path, 'test.pkl'), 'rb') as f:
        idx_test = pickle.load(f)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    if not sparse:
        adj = adj.todense()
        d = {}
        for x in range(0, adj.shape[0]):
            for y in range(0, adj.shape[1]):
                if adj[x, y] == 0:
                    pass
                elif adj[x, y] >= 1:
                    adj[x, y] = 1
                    d[x] = d.get(x, 0) + 1
                else:
                    print(x, y, adj[x, y], 'error')
        if adj_norm:
            for x in range(0, adj.shape[0]):
                for y in range(0, adj.shape[1]):
                    if adj[x, y] > 0:
                        adj[x, y] = adj[x, y] / (math.sqrt(d[x] * d[y]))
        adj = torch.FloatTensor(np.array(adj))
    else:
        N = adj.shape[0]
        us, vs, ws = [], [], []
        d = {}
        for i in range(0, N):
            for j in range(adj.indptr[i], adj.indptr[i + 1]):
                d[i] = d.get(i, 0) + 1
                us.append(i)
                vs.append(adj.indices[j])
                w = adj.data[j]
                if w >= 1:
                    w = 1
                elif w < 0:
                    print(w, 'error')
                ws.append(w)
        if adj_norm:
            for i in range(0, len(us)):
                ws[i] = ws[i] / (math.sqrt(d[us[i]] * d[vs[i]]))
        index = torch.LongTensor([us, vs])
        value = torch.FloatTensor(ws)
        adj = torch.sparse.FloatTensor(index, value, torch.Size([N, N]))

    # binary
    for i, v in enumerate(features.data):
        if v > 0:
            features.data[i] = 1
        else:
            print('error', v)
            features.data[i] = 0

    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.array(labels))
    labels_onehot = torch.FloatTensor(np.array(labels_onehot))
    idx_train = torch.LongTensor(np.array(idx_train))
    idx_val = torch.LongTensor(np.array(idx_val))
    idx_test = torch.LongTensor(np.array(idx_test))

    return adj, features, labels, labels_onehot, idx_train, idx_val, idx_test


def read_features(file):
    xlist = []
    ylist = []
    vlist = []
    with open(file, 'r') as f:
        for line in f.readlines():
            x, s = line.strip().split('\t')
            for i in s.split(' '):
                y, v = i.split(':')
                xlist.append(int(x))
                ylist.append(int(y))
                vlist.append(float(v))
    N = max(xlist) + 1
    Y = max(ylist) + 1
    features = sp.csr_matrix((vlist, (xlist, ylist)), shape=(N, Y),
                             dtype=np.float32)
    features = normalize_features(features)
    features = torch.FloatTensor(np.array(features.todense()))
    return features


def read_idx(file):
    a = []
    with open(file, 'r') as f:
        for line in f.readlines():
            a.append(line.strip())
    a = torch.LongTensor(np.array(a, dtype=np.int32))
    return a


def read_labels(file, N):
    labels = [-1] * N
    with open(file, 'r') as f:
        for line in f.readlines():
            x, l = line.strip().split('\t')
            labels[int(x)] = int(l)

    labels_onehot = labels_to_onehot(labels)

    return torch.LongTensor(np.array(labels)), \
           torch.FloatTensor(np.array(labels_onehot))


def labels_to_onehot(labels):
    nclass = max(labels) + 1
    labels_onehot = []
    for l in labels:
        onehot = [0] * nclass
        if l >= 0:
            onehot[l] = 1
        labels_onehot.append(onehot)
    return labels_onehot


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
