#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:58:34 2018

@author: harry
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence 
from torch.nn.utils.rnn import pad_packed_sequence

from hparam import hparam as hp
from utils import get_centroids, get_cossim, calc_loss, get_distance, calc_contrast_loss, calc_loss_distance
from model_vc import Generator

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

    def freeze(self):
        for model in (self.LSTM_stack, self.projection):
            for param in model.parameters():
                param.requires_grad = False

    # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

        
    def model(self, padded_X, length_X):
        # https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        packed_Xs = pack_padded_sequence(padded_X, length_X,
                                         batch_first=True, enforce_sorted=False)
        self.LSTM_stack.flatten_parameters()
        X, _ = self.LSTM_stack(packed_Xs.float()) #(batch, frames, n_mels)
        X, _ = pad_packed_sequence(X, batch_first=True)
        #only use last frame
        #https://discuss.pytorch.org/t/how-to-extract-the-last-output-of-lstms/15519
        arange = torch.arange(0, padded_X.size(0), dtype=torch.int64)
        X = X[arange, length_X-1]
        X = self.projection(X.float())
        X = X / torch.norm(X, dim=1).unsqueeze(1)
        return X

    def forward(self, padded_Xs, length_Xs, **kwags):
        N, M, T = padded_Xs.shape[:3]
        padded_X = padded_Xs.reshape(N*M, T, -1)
        length_X = length_Xs.reshape(N*M)
        S_X = self.model(padded_X, length_X)
        # Restore the original shape
        S_X = S_X.reshape(N, M, -1)
        return {"X": padded_X,
                "S_X":S_X}

class GE2ELoss(nn.Module):
    
    def __init__(self, device):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0).to(device), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0).to(device), requires_grad=True)
        self.device = device
        
    def forward(self, embeddings, **kwargs):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w*cossim.to(self.device) + self.b
        loss, _ = calc_loss(sim_matrix)
        return {"ge2e":loss}

class ContrastLoss(GE2ELoss):
    
    def __init__(self, device):
        super(ContrastLoss, self).__init__(device)
        
    def forward(self, embeddings, **kwargs):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w*cossim.to(self.device) + self.b
        loss, _ = calc_contrast_loss(sim_matrix)
        return {"ge2e":loss}

class SpeechEmbedderScaledNorm(SpeechEmbedder):
    
    def __init__(self):
        super(SpeechEmbedderScaledNorm, self).__init__()    
        self.scale = nn.Linear(hp.model.proj, 1)
        
    def model(self, padded_X, length_X):
        # https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        packed_Xs = pack_padded_sequence(padded_X, length_X,
                                         batch_first=True, enforce_sorted=False)
        self.LSTM_stack.flatten_parameters()
        X, _ = self.LSTM_stack(packed_Xs.float()) #(batch, frames, n_mels)
        X, _ = pad_packed_sequence(X, batch_first=True)
            
        #only use last frame
        #https://discuss.pytorch.org/t/how-to-extract-the-last-output-of-lstms/15519
        arange = torch.arange(0, padded_X.size(0), dtype=torch.int64)
        X = X[arange, length_X-1]
        X = self.projection(X.float())
        s = F.softplus(self.scale(X))
        X = F.normalize(X, dim=1)
        return X/s

    def forward(self, padded_Xs, length_Xs, **kwags):
        N, M, T = padded_Xs.shape[:3]
        padded_X = padded_Xs.reshape(N*M, T, -1)
        length_X = length_Xs.reshape(N*M)
        S_X = self.model(padded_X, length_X)
        # Restore the original shape
        S_X = S_X.reshape(N, M, -1)
        return {"X": padded_X,
                "S_X":S_X}

class LpDistanceLoss(nn.Module):
    
    def __init__(self, device, p=2):
        super(LpDistanceLoss, self).__init__()
        self.device = device
        self.p = p
        
    def forward(self, embeddings, disc=False, dbi=False):
        centroids = get_centroids(embeddings)
        distance = get_distance(embeddings, centroids, self.p)
        d_matrix = distance.to(self.device)
        loss, _ = calc_loss_distance(d_matrix)
        return {"ge2e":loss}


class SpeechEmbedderAutoVC(SpeechEmbedderScaledNorm):
    def __init__(self):
        super(SpeechEmbedderAutoVC, self).__init__()    
        self.dim_neck = hp.autovc.dim_neck
        self.dim_pre = hp.autovc.dim_pre
        self.freq = hp.autovc.freq

        self.G = Generator(self.dim_neck, self.dim_pre, self.freq)
    
    def train(self, mode=True):
        self.G.train(mode)
        return super(SpeechEmbedderAutoVC, self).train(mode)

    def to(self, device):
        self.G = self.G.to(device)
        return super(SpeechEmbedderAutoVC, self).to(device)

    def same_style(self, padded_X, length_X, S_X):
        N, M, D_S = S_X.shape
        S_X = S_X.reshape(N*M, D_S)

        _, T, D = padded_X.shape
        X_tilda, X_hat, C_X = self.G(padded_X, length_X, S_X, S_X)
        C_hat = self.G(X_hat, length_X, S_X, None)
        # max length changes
        T_C = C_hat.size(1)
        D_C = C_hat.size(2)
        C_X = C_X[:,:T_C]
        # Restore the original shape
        padded_X = padded_X.reshape(N, M, T, D)
        length_X = length_X.reshape(N, M)
        X_tilda = X_tilda.reshape(N, M, T, D)
        X_hat = X_hat.reshape(N, M, T, D)
        C_X = C_X.reshape(N, M, T_C, D_C)
        C_hat = C_hat.reshape(N, M, T_C, D_C)
        return {
                "X": padded_X,
                "L": length_X,
                "X_tilda":X_tilda, 
                "X_hat":X_hat, 
                "C_X":C_X, 
                "C_hat":C_hat, 
                }

    def forward(self, padded_Xs, length_Xs, autovc=False):
        N, M, T, D = padded_Xs.shape
        padded_X = padded_Xs.reshape(N*M, T, D)
        length_X = length_Xs.reshape(N*M)
        # max length changes when testing
        packed_X = pack_padded_sequence(padded_X, length_X, batch_first=True, enforce_sorted=False)
        padded_X, length_X = pad_packed_sequence(packed_X, batch_first=True)
        S_X = self.model(padded_X, length_X)
        D_S = S_X.size(1)
        # Restore the original shape
        S_X = S_X.reshape(N, M, D_S)
        ret = {"S_X":S_X}
        if autovc:
            ret_aug = self.same_style(padded_X, length_X, S_X)
            ret.update(ret_aug)
        return ret

class SpeechEmbedderAutoVCv2(SpeechEmbedderAutoVC):
    def __init__(self, K):
        super(SpeechEmbedderAutoVCv2, self).__init__()    
        self.K = K
    
    def diff_style(self, padded_Xs, length_Xs, S_X, S_trg):
        N, M, T, D = padded_Xs.shape
        assert N >= 2
        K = self.K
        assert K <= N*M
        # S_X: (N, M, D_S)
        assert len(S_X.shape) == 3
        D_S = S_X.size(2)
        # ensure using different speakers for hallucination
        # pick K different contents only from other speaker 
        arange = torch.arange(0, N, dtype=torch.int64)
        pick_spk = []
        for offset in range(N):
            others = arange[(arange+offset)%N][1:]
            perm = np.random.choice(others, K, replace=True)
            pick_spk.append(perm)
        # make indexing easier
        pick_spk = torch.as_tensor(pick_spk).reshape(N*K)
        # transfer K different utterance for each speaker
        # assume already shuffled when loading
        pick_utt = np.random.choice(M, N*K, replace=True)
        padded_X_K = padded_Xs[pick_spk, pick_utt]
        length_X_K = length_Xs[pick_spk, pick_utt]
        S_X_K = S_X[pick_spk, pick_utt]
        # max length changes after picking
        packed_X = pack_padded_sequence(padded_X_K, length_X_K, batch_first=True, enforce_sorted=False)
        padded_X_K, length_X_K = pad_packed_sequence(packed_X, batch_first=True)
        # An AutoVC Variant
        X_tilda, X_hat, C_X = self.G(padded_X_K, length_X_K, S_X_K, S_trg)
        C_hat = self.G.encoder_fn(X_hat, length_X_K, S_trg)
        S_hat = self.model(X_hat, length_X_K)
        # max length changes
        T = padded_X_K.size(1)
        T_C = C_hat.size(1)
        D_C = C_hat.size(2)
        C_X = C_X[:,:T_C]
        # Restore the original shape
        padded_X_K = padded_X_K.reshape(N, K, T, D)
        length_X_K = length_X_K.reshape(N, K)
        X_tilda = X_tilda.reshape(N, K, T, D)
        X_hat = X_hat.reshape(N, K, T, D)
        C_X = C_X.reshape(N, K, T_C, D_C)
        C_hat = C_hat.reshape(N, K, T_C, D_C)
        S_hat = S_hat.reshape(N, K, D_S)
        S_X_K = S_X_K.reshape(N, K, D_S)
        return {
                "X": padded_X_K,
                "L": length_X_K,
                "X_tilda":X_tilda, 
                "X_hat":X_hat, 
                "C_X":C_X, 
                "C_hat":C_hat, 
                "S_hat":S_hat, 
                "S_neg":S_X_K, # it should be different from S_hat
                }

    def forward(self, padded_Xs, length_Xs, autovc=False):
        N, M, T, D = padded_Xs.shape
        K = self.K
        padded_X = padded_Xs.reshape(N*M, T, D)
        length_X = length_Xs.reshape(N*M)
        # max length changes when testing
        packed_X = pack_padded_sequence(padded_X, length_X, batch_first=True, enforce_sorted=False)
        padded_X, length_X = pad_packed_sequence(packed_X, batch_first=True)
        S_X = self.model(padded_X, length_X)
        D_S = S_X.size(1)
        # Restore the original shape (before halluciation)
        S_X = S_X.reshape(N, M, D_S)
        ret = {"S_X":S_X}
        # pick different utterance embedding if possible
        arange = torch.arange(0, N*K, dtype=torch.int64)//K
        pick_utt = np.random.choice(M, N*K, replace=True)
        S_trg = S_X[arange, pick_utt].reshape(N*K, D_S)
        if not self.training: # just need S_hat
            ret_aug = self.diff_style(padded_Xs, length_Xs, S_X, S_trg)
            ret["S_hat"] = ret_aug["S_hat"]
            return ret
        if autovc:
            ret_aug = self.diff_style(padded_Xs, length_Xs, S_X, S_trg)
            ret.update(ret_aug)
        return ret

    
def split_spk(N, K):
    '''
    ensure using different speakers for hallucination
    pick K different contents only from other speaker 
    divide by two halves for positive and negative samples 
    '''
    arange = torch.arange(0, N, dtype=torch.int64)
    pick_p1 = []
    pick_p2 = []
    for offset in range(N):
        others = arange[(arange+offset)%N][1:]
        # for the p1itive speakers
        perm = np.random.choice(others[:(N-1)//2], K, replace=True)
        pick_p1.append(perm)
        # for the p2ative speakers
        perm = np.random.choice(others[(N-1)//2:], K, replace=True)
        pick_p2.append(perm)
    # transfer K different utterance for each speaker
    # make indexing easier
    pick_p1 = torch.as_tensor(pick_p1).reshape(N*K)
    pick_p2 = torch.as_tensor(pick_p2).reshape(N*K)
    return pick_p1, pick_p2

class SpeechEmbedderAutoVCv3(SpeechEmbedderAutoVCv2):
    def __init__(self, K):
        super(SpeechEmbedderAutoVCv3, self).__init__(K)    
    
    def forward(self, padded_Xs, length_Xs, autovc=False):
        N, M, T, D = padded_Xs.shape
        K = self.K
        padded_X = padded_Xs.reshape(N*M, T, D)
        length_X = length_Xs.reshape(N*M)
        # max length changes when testing
        packed_X = pack_padded_sequence(padded_X, length_X, batch_first=True, enforce_sorted=False)
        padded_X, length_X = pad_packed_sequence(packed_X, batch_first=True)
        T = padded_X.size(1)
        S_X = self.model(padded_X, length_X)
        # Restore the original shape
        D_S = S_X.size(-1)
        S_X = S_X.reshape(N, M, D_S)
        ret = {"S_X":S_X}
        # pick different utterance embedding if possible
        arange = torch.arange(0, N*K, dtype=torch.int64)//K
        pick_utt = np.random.choice(M, N*K, replace=True)
        S_trg = S_X[arange, pick_utt].reshape(N*K, D_S)
        if not self.training: # just need S_hat
            ret_aug = self.diff_style(padded_Xs, length_Xs, S_X, S_trg)
            ret["S_hat"] = ret_aug["S_hat"]
            return ret
        if autovc:
            ret_aug = self.diff_style(padded_Xs, length_Xs, S_X, S_trg)
            ret.update(ret_aug)
            # restore shape
            S_trg = S_trg.reshape(N, K, D_S)
            # abuse S_hat
            ret["S_anc"] = ret_aug["S_hat"]
            ret["S_pos"] = S_trg
        return ret

class SpeechEmbedderAutoVCv4(SpeechEmbedderAutoVCv2):
    def __init__(self, K):
        super(SpeechEmbedderAutoVCv4, self).__init__(K)    

    def diff_content_same_style(self, padded_Xs, length_Xs, S_org, S_trg):
        """
        for computing same style
        """
        N, M, T, D = padded_Xs.shape
        assert N >= 3
        K = self.K
        assert 2*K <= (N-1)*M
        pick_V = split_spk(N, K)
        S_X = [None] * len(pick_V)
        C_X = [None] * len(pick_V)
        C_V = [None] * len(pick_V)
        S_V = [None] * len(pick_V)
        D_S = S_org.size(-1)
        for i in range(len(pick_V)):
            # assume already shuffled when loading
            pick_utt = np.random.choice(M, N*K, replace=True)
            padded_X_K = padded_Xs[pick_V[i], pick_utt]
            length_X_K = length_Xs[pick_V[i], pick_utt]
            S_X[i] = S_org[pick_V[i], pick_utt].reshape(N*K, D_S)
            # max length changes after picking
            packed_X = pack_padded_sequence(padded_X_K, length_X_K, batch_first=True, enforce_sorted=False)
            padded_X_K, length_X_K = pad_packed_sequence(packed_X, batch_first=True)
            # An AutoVC Variant
            _, X_hat, C_X[i] = self.G(padded_X_K, length_X_K, S_X[i], S_trg)
            C_V[i] = self.G.encoder_fn(X_hat, length_X_K, S_trg)
            S_V[i] = self.model(X_hat, length_X_K)
            # Restore the original shape
            S_X[i] = S_X[i].reshape(N, K, D_S)
            S_V[i] = S_V[i].reshape(N, K, D_S)

        T_C = min(C_V[i].size(1) for i in range(len(pick_V)))
        D_C = C_V[0].size(2)
        for i in range(len(pick_V)):
            # Restore the original shape
            # max length changes after picking
            C_X[i] = C_X[i][:, :T_C]
            C_V[i] = C_V[i][:, :T_C]
            C_X[i] = C_X[i].reshape(N, K, T_C, D_C)
            C_V[i] = C_V[i].reshape(N, K, T_C, D_C)
        return {
                "S_X_1":S_X[0], "S_X_2":S_X[1], 
                "S_V_1":S_V[0], "S_V_2":S_V[1], 
                "C_X_1":C_X[0], "C_X_2":C_X[1], 
                "C_V_1":C_V[0], "C_V_2":C_V[1], 
                }

    def forward(self, padded_Xs, length_Xs, autovc=False):
        N, M, T, D = padded_Xs.shape
        K = self.K
        padded_X = padded_Xs.reshape(N*M, T, D)
        length_X = length_Xs.reshape(N*M)
        # max length changes when testing
        packed_X = pack_padded_sequence(padded_X, length_X, batch_first=True, enforce_sorted=False)
        padded_X, length_X = pad_packed_sequence(packed_X, batch_first=True)
        T = padded_X.size(1)
        S_X = self.model(padded_X, length_X)
        D_S = S_X.size(-1)
        # Restore the original shape
        S_X = S_X.reshape(N, M, D_S)
        ret = {"S_X":S_X}
        # pick different utterance embedding if possible
        arange = torch.arange(0, N*K, dtype=torch.int64)//K
        pick_utt = np.random.choice(M, N*K, replace=True)
        S_trg = S_X[arange, pick_utt].reshape(N*K, D_S)
        if not self.training: # just need S_hat
            ret_aug = self.diff_style(padded_Xs, length_Xs, S_X, S_trg)
            ret["S_hat"] = ret_aug["S_hat"]
            return ret
        if autovc:
            ret_aug = self.diff_style(padded_Xs, length_Xs, S_X, S_trg)
            ret_style_v = self.diff_content_same_style(padded_Xs, length_Xs, S_X, S_trg)
            ret.update(ret_aug)
            ret.update(ret_style_v)
            # abuse C_X
            C_X = ret_aug["C_X"]
            T_C = min(ret_style_v["C_X_1"].size(-1), ret_style_v["C_X_2"].size(-1))
            ret["C_neg"] = C_X[:,:,:T_C]
        return ret

class SpeechEmbedderAutoVCv5(SpeechEmbedderAutoVCv4):
    def __init__(self, K):
        super(SpeechEmbedderAutoVCv5, self).__init__(K)    
    
    def forward(self, padded_Xs, length_Xs, autovc=False):
        N, M, T, D = padded_Xs.shape
        K = self.K
        padded_X = padded_Xs.reshape(N*M, T, D)
        length_X = length_Xs.reshape(N*M)
        # max length changes when testing
        packed_X = pack_padded_sequence(padded_X, length_X, batch_first=True, enforce_sorted=False)
        padded_X, length_X = pad_packed_sequence(packed_X, batch_first=True)
        T = padded_X.size(1)
        S_X = self.model(padded_X, length_X)
        # Restore the original shape
        D_S = S_X.size(-1)
        S_X = S_X.reshape(N, M, D_S)
        ret = {"S_X":S_X}
        # pick different utterance embedding if possible
        arange = torch.arange(0, N*K, dtype=torch.int64)//K
        pick_utt = np.random.choice(M, N*K, replace=True)
        S_trg = S_X[arange, pick_utt].reshape(N*K, D_S)
        if not self.training: # just need S_hat
            ret_aug = self.diff_style(padded_Xs, length_Xs, S_X, S_trg)
            ret["S_hat"] = ret_aug["S_hat"]
            return ret
        if autovc:
            ret_aug = self.diff_style(padded_Xs, length_Xs, S_X, S_trg)
            ret_style_v = self.diff_content_same_style(padded_Xs, length_Xs, S_X, S_trg)
            ret.update(ret_aug)
            ret.update(ret_style_v)
            # restore shape
            S_trg = S_trg.reshape(N, K, D_S)
            # abuse S_hat
            ret["S_anc"] = ret_aug["S_hat"]
            ret["S_pos"] = S_trg
            # abuse C_X
            C_X = ret_aug["C_X"]
            T_C = min(ret_style_v["C_X_1"].size(-1), ret_style_v["C_X_2"].size(-1))
            ret["C_neg"] = C_X[:,:,:T_C]
        return ret
