from __future__ import print_function
import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from sklearn import metrics

sys.path.append('%s/../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import Classifier
from DGCNN_embedding import DGCNN
from util import cmd_args, load_data
from pytorch_util import weights_init


# rewrite MLPs to return more scores
class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size, with_dropout=False):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)
        pred = self.h2_weights(h1)[:, 0]

        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            mae = mae.cpu().detach()
            return pred, mae, mse
        else:
            return pred


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, with_dropout=False):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        self.with_dropout = with_dropout

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)

        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y-1) # convert ratings 1, 2, ... 5 to classes 0, 1, ... 4
            loss = F.nll_loss(logits, y)

            #pred = logits.data.max(1, keepdim=True)[1]
            #acc = pred.eq(y.data.view_as(pred)).cpu().sum().item() / float(y.size()[0])

            probs = logits.exp()
            scores = torch.FloatTensor(range(1, logits.shape[1]+1)).to(logits.device)
            pred_continuous = (probs * scores).sum(1)

            y = y.type(torch.FloatTensor).to(y.device)
            mse = F.mse_loss(pred_continuous, y)
            mae = F.l1_loss(pred_continuous, y)
            mse = mse.cpu().detach()
            mae = mae.cpu().detach()
            return pred_continuous, loss, mae, mse
        else:
            return logits


def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        if classifier.regression:
            pred, mae, loss = classifier(batch_graph)
        else:
            pred, loss, mae, mse = classifier(batch_graph)
        all_scores.append(pred.cpu().detach())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #loss = loss.data.cpu().detach().numpy()
        loss = loss.item()
        if classifier.regression:
            pbar.set_description('RMSE_loss: %0.5f MAE_loss: %0.5f' % (np.sqrt(loss), mae) )
            total_loss.append( np.array([loss, mae]) * len(selected_idx))
        else:
            pbar.set_description('CE_loss: %0.5f RMSE_loss: %0.5f MAE_loss: %0.5f' % (loss, mse.sqrt(), mae) )
            total_loss.append( np.array([mse, mae]) * len(selected_idx))


        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    avg_loss[0] =np.sqrt(avg_loss[0])
    all_scores = torch.cat(all_scores).cpu().numpy()
    
    np.savetxt('test_scores.txt', all_scores)  # output test predictions
    
    return avg_loss


