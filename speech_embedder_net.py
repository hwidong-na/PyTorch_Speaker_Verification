#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:58:34 2018

@author: harry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence 
from torch.nn.utils.rnn import pad_packed_sequence

from hparam import hparam as hp
from utils import get_centroids, get_cossim, calc_loss, get_distance, calc_contrast_loss, calc_loss_euclidean

class SpeechEmbedder(nn.Module):
    
    def __init__(self):
        super(SpeechEmbedder, self).__init__()    
        self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden, num_layers=hp.model.num_layer, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
          if 'bias' in name:
             nn.init.constant_(param, 0.0)
          elif 'weight' in name:
             nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hp.model.hidden, hp.model.proj)
        
    def forward(self, x):
        x, _ = self.LSTM_stack(x.float()) #(batch, frames, n_mels)
        #only use last frame
        x = x[:,x.size(1)-1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x

class GE2ELoss(nn.Module):
    
    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device
        
    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w*cossim.to(self.device) + self.b
        loss, _ = calc_loss(sim_matrix)
        return loss

class ContrastLoss(GE2ELoss):
    
    def __init__(self, device):
        super(ContrastLoss, self).__init__(device)
        
    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w*cossim.to(self.device) + self.b
        loss, _ = calc_contrast_loss(sim_matrix)
        return loss

class SpeechEmbedderV2(SpeechEmbedder):
    
    def __init__(self):
        super(SpeechEmbedderV2, self).__init__()    
        self.scale = nn.Linear(hp.model.proj, 1)
        
    def forward(self, x):
        x, _ = self.LSTM_stack(x.float()) #(batch, frames, n_mels)
        #only use last frame
        x = x[:,x.size(1)-1]
        x = self.projection(x.float())
        s = F.softplus(self.scale(x))
        x = F.normalize(x, p=2, dim=1)
        return x/s

class EuclideanDistanceLoss(nn.Module):
    
    def __init__(self, device):
        super(EuclideanDistanceLoss, self).__init__()
        self.device = device
        
    def forward(self, embeddings):
        centroids = get_centroids(embeddings)
        distance = get_distance(embeddings, centroids)
        d_matrix = distance.to(self.device)
        loss, _ = calc_loss_euclidean(d_matrix)
        return loss

class SpeechEmbedderV3(SpeechEmbedderV2):
    
    def forward(self, X):
        # https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        padded_X, length_X = X
        packed_Xs = pack_padded_sequence(padded_X, length_X,
                                         batch_first=True, enforce_sorted=False)
        X, _ = self.LSTM_stack(packed_Xs.float()) #(batch, frames, n_mels)
        X, _ = pad_packed_sequence(X, batch_first=True)
            
        #only use last frame
        #https://discuss.pytorch.org/t/how-to-extract-the-last-output-of-lstms/15519
        arange = torch.arange(0, padded_X.size(0), dtype=torch.int64)
        X = X[arange, length_X-1]
        X = self.projection(X.float())
        s = F.softplus(self.scale(X))
        X = F.normalize(X, p=2, dim=1)
        return X/s

class AutoVCLoss(EuclideanDistanceLoss):

    def __init__(self, device):
        super(AutoVCLoss, self).__init__(device)
