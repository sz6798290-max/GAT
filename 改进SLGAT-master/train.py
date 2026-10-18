# coding=utf-8

import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from trainer import Trainer
from loader import load_gnn_data, load_gcn_graph
from models import SLGAT, SpSLGAT
import utils

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--pre-epochs', type=int, default=200, help='Number of epochs to pre-train.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--class-hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--input-dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight-dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--data', type=str, default='cora', help='Dataset name')
parser.add_argument('--file', type=str, default=None, help='Result file')
parser.add_argument('--cuda-id', type=int, default=None, help="CUDA id")
parser.add_argument('--model', type=str, default='SLGAT', help="Model name")
parser.add_argument('--tau', type=float, default=0.1, help="Tau")

# 你公式里的超参数
parser.add_argument('--alpha', type=float, default=0.5, help="Alpha weight")
parser.add_argument('--beta', type=float, default=0.5, help="Beta weight")
parser.add_argument('--T_p', type=float, default=0.1, help="Temperature for soft targets")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    if args.cuda_id is not None:
        torch.cuda.set_device(args.cuda_id)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.determinstic = True
os.environ['PYTHONHASHSEED'] = str(args.seed)

# Load data
adj, features, labels, labels_onehot, idx_train, idx_val, idx_test = \
    load_gnn_data(args.data, sparse=True)

num_class = int(labels.max()) + 1
num_feature = features.size()[1]
num_nodes = features.size()[0]

idx_all = torch.LongTensor([i for i in range(num_nodes)])
idx_unlabeled = torch.LongTensor([i for i in range(idx_train.size()[0], num_nodes)])

print('# seed:', args.seed)
print('# nodes:', num_nodes)
print('# features:', num_feature)
print('# classes:', num_class)
print('# model:', args.model)

if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    labels_onehot = labels_onehot.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_all = idx_all.cuda()
    idx_unlabeled = idx_unlabeled.cuda()

# Model and optimizer
if args.model == 'SLGAT':
    gcn_adj = load_gcn_graph(args.data)
    if args.cuda:
        gcn_adj = gcn_adj.cuda()
    biased_adj = utils.adj_bias_normalize(adj.to_dense())
    model = SLGAT(num_feature=num_feature, hidden_dim=args.hidden, num_class=num_class,
                  class_hidden=args.class_hidden, adj=biased_adj, gcn_adj=gcn_adj.to_dense(),
                  input_dropout=args.input_dropout, dropout=args.dropout,
                  weight_dropout=args.weight_dropout)
    if args.cuda:
        model.cuda()
    trainer = Trainer(args, model)
elif args.model == 'SpSLGAT':
    model = SpSLGAT(num_feature=num_feature, hidden_dim=args.hidden, num_class=num_class,
                    class_hidden=args.class_hidden, adj=adj.to_dense(), gcn_adj=adj,
                    input_dropout=args.input_dropout, dropout=args.dropout,
                    weight_dropout=args.weight_dropout)
    if args.cuda:
        model.cuda()
    trainer = Trainer(args, model)
else:
    raise ValueError('invalid model name: {}'.format(args.model))

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

results = []

def get_accuracy(results):
    best_epoch = 0
    best_dev, acc_test = 0.0, 0.0
    for i, (d, t) in enumerate(results):
        if d > best_dev:
            best_epoch = i
            best_dev, acc_test = d, t
    return best_epoch, acc_test


def train():
    pre_epochs = args.pre_epochs
    epochs = args.epochs

    # 预计算带自环的邻接矩阵，用于后续计算 d_i
    adj_with_self_loop = adj + torch.sparse_coo_tensor(
        torch.arange(num_nodes).unsqueeze(0).repeat(2, 1).to(adj.device),
        torch.ones(num_nodes).to(adj.device),
        (num_nodes, num_nodes)
    )
    rowsum = torch.sparse.sum(adj_with_self_loop, dim=1).to_dense()
    deg_inv = 1.0 / rowsum

    # 1. 预训练阶段
    if pre_epochs > 0:
        for i in range(pre_epochs):
            model.train()
            optimizer.zero_grad()
            
            output = model(features)
            loss = F.cross_entropy(output[idx_train], labels[idx_train])
            
            loss.backward()
            optimizer.step()
            
            train_loss, _, train_acc = trainer.evaluate(features, labels, idx_train)
            _, _, val_acc = trainer.evaluate(features, labels, idx_val)
            _, _, test_acc = trainer.evaluate(features, labels, idx_test)

            print('pre-train Epoch: {:04d}'.format(i),
                  'train loss: {:.4f}'.format(train_loss),
                  'train acc: {:.4f}'.format(train_acc),
                  'val acc: {:.4f}'.format(val_acc),
                  'test acc: {:.4f}'.format(test_acc))

    # 2. 自训练阶段（你的公式生效）
    for i in range(epochs):
        t = time.time()

        model.train()
        optimizer.zero_grad()
        
        output = model(features)
        probs = F.softmax(output, dim=1)
        log_probs = F.log_softmax(output, dim=1)
        
        # --- 监督损失 L_sup ---
        loss_sup = F.cross_entropy(output[idx_train], labels[idx_train])
        
        # --- 计算 d_i：邻居标签分布 ---
        # 初始化标签信号 Y_signal
        Y_signal = probs.detach().clone()
        Y_signal[idx_train] = labels_onehot[idx_train] # 训练集用真实标签覆盖
        
        # 聚合邻居 (D^{-1} * A * Y)
        d_i = torch.sparse.mm(adj_with_self_loop, Y_signal)
        d_i = d_i * deg_inv.unsqueeze(1)  # 此时 d_i 维度为 [N, C]
        
        # --- 计算 p_c (类别中心) ---
        p_c = torch.zeros((num_class, num_class), device=d_i.device)
        for c in range(num_class):
            mask_c = torch.zeros(num_nodes, dtype=torch.bool, device=d_i.device)
            mask_c[idx_train] = (labels[idx_train] == c)
            if mask_c.sum() > 0:
                p_c[c] = d_i[mask_c].mean(dim=0)
                
        p_bar = p_c.mean(dim=0, keepdim=True)
        p_c_tilde = p_c - p_bar
        
        # 计算余弦相似度 [N, C]
        d_i_norm = F.normalize(d_i, p=2, dim=1)
        p_c_tilde_norm = F.normalize(p_c_tilde, p=2, dim=1)
        cos_sim = torch.mm(d_i_norm, p_c_tilde_norm.t())
        
        pseudo_labels = torch.argmax(probs, dim=1)
        
        A_i = cos_sim[torch.arange(num_nodes), pseudo_labels]
        cos_sim_no_self = cos_sim.clone()
        cos_sim_no_self[torch.arange(num_nodes), pseudo_labels] = -1e9
        B_i, _ = torch.max(cos_sim_no_self, dim=1)
        
        t_i = A_i - B_i
        t_i_norm = (t_i + 1) / 2
        c_i = torch.max(probs, dim=1)[0]
        
        # f_i 综合得分
        f_i = args.alpha * c_i + args.beta * t_i_norm
        
        # 筛选集合 M (不是训练集 且 f_i >= tau)
        mask_M = torch.ones(num_nodes, dtype=torch.bool, device=d_i.device)
        mask_M[idx_train] = False
        mask_M = mask_M & (f_i >= args.tau)
        
        # --- 无监督损失 L_unsup ---
        if mask_M.sum() > 0:
            q_i = F.softmax(log_probs / args.T_p, dim=1)
            loss_unsup = -torch.mean(torch.sum(q_i[mask_M] * log_probs[mask_M], dim=1))
        else:
            loss_unsup = torch.tensor(0.0, device=d_i.device)
            
        # --- 总损失 ---
        total_loss = loss_sup + loss_unsup
        
        total_loss.backward()
        optimizer.step()

        train_loss, _, train_acc = trainer.evaluate(features, labels, idx_train)
        val_loss, _, val_acc = trainer.evaluate(features, labels, idx_val)
        _, _, test_acc = trainer.evaluate(features, labels, idx_test)
        results.append((val_acc, test_acc))

        print('Epoch: {:04d}'.format(i),
              'loss: {:.4f}'.format(total_loss.item()),
              'train acc: {:.4f}'.format(train_acc),
              'val acc: {:.4f}'.format(val_acc),
              'test acc: {:.4f}'.format(test_acc),
              'time: {:.4f}s'.format(time.time() - t))


t_total = time.time()
train()

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

best_epoch, best_acc = get_accuracy(results)
print('{}th epoch: {:.4f}'.format(best_epoch, best_acc))

if args.file:
    with open(args.file, 'a') as f:
        f.write("{:.4f}\n".format(best_acc))