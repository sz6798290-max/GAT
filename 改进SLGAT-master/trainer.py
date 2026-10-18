# coding=utf-8

import torch
from torch import nn, optim


class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        if args.cuda:
            self.criterion.cuda()

    def update_soft(self, inputs, target, idx, idx_unlabeled=None, coef=1.0):
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        logits = torch.log_softmax(logits, dim=1)
        loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=1))

        if idx_unlabeled is not None:
            loss_u = -torch.mean(torch.sum(target[idx_unlabeled] * logits[idx_unlabeled], dim=1))
            loss += coef * loss_u

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, inputs, target, idx):
        self.model.eval()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        preds = torch.max(logits[idx], dim=1)[1]
        correct = preds.eq(target[idx]).double()
        accuracy = correct.sum() / idx.size(0)

        return loss.item(), preds, accuracy.item()

    def predict(self, inputs, tau=1):
        self.model.eval()
        logits = self.model(inputs)
        logits = torch.softmax(logits / tau, dim=-1).detach()
        return logits
