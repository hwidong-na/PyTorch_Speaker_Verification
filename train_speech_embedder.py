#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2018

@author: harry
"""

import os
import random
import time
import re
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn

from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from data_load import SpeakerDatasetSpectrogram
from speech_embedder_net import *
from utils import get_centroids, get_cosdiff, get_distance_forall

class Reporter():
    def __init__(self, reports, p=2, delimiter=":"):
        self.keys = reports.split(delimiter)
        self.p = p

    def __call__(self, embeddings):
        ret = {}
        centroids = get_centroids(embeddings)
        if "v_inter" in self.keys:
            N, M, D = embeddings.shape
            c_k = centroids.unsqueeze(0).repeat(N,1,1)
            c_m = centroids.unsqueeze(1).repeat(1,N,1)
            v_inter = torch.norm(c_k - c_m, dim=2, p=self.p).pow(self.p).sum() / N / (N-1)
            ret.update({
                "v_inter":v_inter,
            })
        if "v_intra" in self.keys:
            N, M, D = embeddings.shape
            c_km = centroids.unsqueeze(1).repeat(1,M,1)
            v_intra = torch.norm(embeddings - c_km, dim=2, p=self.p).pow(self.p).sum() / N / M
            ret.update({
                "v_intra":v_intra,
            })
        if "dbi" in self.keys:
            N, M, D = embeddings.shape
            c_k = centroids.unsqueeze(0).repeat(N,1,1)
            c_m = centroids.unsqueeze(1).repeat(1,N,1)
            I = list(range(N))
            M_IJ = torch.norm(c_k - c_m, p=self.p, dim=2)
            M_IJ[I,I] = 1 # prevent divide by zero
            c_km = centroids.unsqueeze(1).repeat(1,M,1)
            S_I = torch.norm(embeddings - c_km, p=self.p, dim=2).mean(dim=1)
            S_J = S_I.unsqueeze(0).repeat(N,1)
            R_IJ = S_I.unsqueeze(1).repeat(1,N) + S_J / M_IJ
            R_IJ[I,I] = 0
            D_I, _ = R_IJ.max(dim=1)
            ret.update({
                "dbi":D_I.mean()
            })
        return ret

def custom_collate(Xs):
    n_dim = hp.data.nmels
    # x.shape = (n_seq, n_dim)
    length_Xs = torch.tensor([tuple(x.shape[0] for x in X) for X in Xs])
    batch_max_len = torch.max(length_Xs)
    padded_Xs = torch.zeros((length_Xs.size(0), length_Xs.size(1), batch_max_len, n_dim))
    
    for i, X in enumerate(Xs):
        lens = length_Xs[i]
        for j, x in enumerate(X):
            padded_Xs[i, j, :lens[j]] = torch.tensor(x[:lens[j]])
    #TODO(hwidongna): sort by length?
    return padded_Xs, length_Xs

def test_epoch(epoch, device, test_loader, embedder_net, reporter_net):
    sum_EER = 0
    total = 0
    for batch_id, (padded_Xs, length_Xs) in enumerate(test_loader): 
        padded_Xs = padded_Xs.to(device)
        length_Xs = length_Xs.to(device)
        # N.B. utterances are shuffled in SpeakerDataset*
        # k-shot test, k is the number enrollment utterences
        assert hp.test.M - hp.test.K > 0
        
        #TODO(hwidongna): generate speaker embeddings
        padded_Xe = padded_Xs[:,:hp.test.K]
        length_Xe = length_Xs[:,:hp.test.K]
        ret = embedder_net(padded_Xe, length_Xe, autovc=True)

        enrollment_embeddings = ret["S_X"]
        if "S_hat" in ret:
            enrollment_embeddings = torch.cat((enrollment_embeddings, ret["S_hat"]), dim=1)
        enrollment_centroids = get_centroids(enrollment_embeddings)
        # separetely run the verification (slower, but required for autovc)
        # do not generate speaker embeddings for the query set
        padded_Xv = padded_Xs[:,hp.test.K:,:,:]
        length_Xv = length_Xs[:,hp.test.K:]
        verification_embeddings = embedder_net(padded_Xv, length_Xv)["S_X"]
        if hp.train.loss_type == "scaled-norm" or hp.train.loss_type.startswith("autovc"):
            d_matrix = get_distance_forall(verification_embeddings, enrollment_centroids, hp.train.norm)
            # because the distance is not bounded, need to normalize
            d_matrix = F.normalize(d_matrix, dim=2)
            threshold_fn = lambda thres: d_matrix < thres
        elif hp.train.loss_type in ("ge2e-softmax", "ge2e-contrast",):
            sim_matrix = get_cosdiff(verification_embeddings, enrollment_centroids)
            threshold_fn = lambda thres: sim_matrix > thres
        else:
            raise Exception("unsupported loss type: %s"%hp.train.loss_type)
        # calculate ERR excluding enrollment
        
        N = verification_embeddings.size(0)
        M = verification_embeddings.size(1)
        # calculating EER
        diff = 1 + 1e-6; EER=1; EER_thresh = 1; EER_FAR=1; EER_FRR=1
        
        for thres in [0.01*i for i in range(100)]:
            matrix_thresh = threshold_fn(thres)
            
            pred = lambda i: matrix_thresh[i].float().sum()
            true = lambda i: matrix_thresh[i,:,i].float().sum()
            false_acceptance = lambda i: pred(i) - true(i)
            FAR = (sum([false_acceptance(i) for i in range(int(N))])
            /(N-1.0)/(float(M))/N)

            false_rejection = lambda i: M - true(i)
            FRR = (sum([false_rejection(i) for i in range(int(N))])
            /(float(M))/N)
            
            # Save threshold when FAR = FRR (=EER)
            if diff> abs(FAR-FRR):
                diff = abs(FAR-FRR)
                EER = (FAR+FRR)/2
                EER_thresh = thres
                EER_FAR = FAR
                EER_FRR = FRR
        sum_EER += EER*N # need to weigh according to N
        total += N
        # human-readable 
        if hp.test.log_interval > 0 and (batch_id + 1) % hp.test.log_interval == 0:
            EER *= 100
            EER_FAR *= 100
            EER_FRR *= 100
            mesg = "\nEER : %.2f (thres:%.2f, FAR:%.2f, FRR:%.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR)
            print(mesg)
            if hp.test.log_file is not None:
                with open(hp.test.log_file,'a') as f:
                    f.write(mesg)
            if reporter_net is not None:
                reports = reporter_net(enrollment_embeddings)
                mesg = "reports"
                for k,v, in reports.items():
                    mesg += ",\t{0}: {1:.4f}".format(k,v)
                mesg += "\n"
                print(mesg)
                if hp.test.log_file is not None:
                    with open(hp.test.log_file,'a') as f:
                        f.write(mesg)
                
    # human-readable 
    avg_EER = sum_EER / total
    avg_EER *= 100
    mesg = "\nEpoch {0} EER : {1:.2f}".format(epoch, avg_EER)
    print(mesg)
    if hp.test.log_file is not None:
        with open(hp.test.log_file,'a') as f:
            f.write(mesg)
    return sum_EER, total

def test():
    device = torch.device(hp.device)
    
    test_dataset = SpeakerDatasetSpectrogram(hp.test.M, hp.data.test_path)
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N,
                              shuffle=True, collate_fn=custom_collate,
                              num_workers=hp.test.num_workers, drop_last=True) 

    reporter_net = Reporter(hp.train.reports, hp.train.norm)
    if hp.train.loss_type.startswith("autovc-v"):
        V = hp.train.loss_type[len("autovc-v"):]
        K = hp.train.K
        embedder_net = eval("SpeechEmbedderAutoVCv{0}({1})".format(V, K)).to(device)
    elif hp.train.loss_type == "autovc":
        embedder_net = SpeechEmbedderAutoVC().to(device)
    elif hp.train.loss_type == "scaled-norm":
        embedder_net = SpeechEmbedderScaledNorm().to(device)
    elif hp.train.loss_type == "ge2e-contrast":
        embedder_net = SpeechEmbedder().to(device)
        reporter_net = None
    elif hp.train.loss_type == "ge2e-softmax":
        embedder_net = SpeechEmbedder().to(device)
        reporter_net = None
    else:
        raise Excpetion("unsupported loss type: %s"%hp.train.loss_type)
    embedder_net.load_state_dict(torch.load(hp.model.model_path))
    embedder_net.eval()
    mesg = "Compute EER using %s"%(hp.train.loss_type)
    print(mesg)
    if hp.test.log_file is not None:
        with open(hp.test.log_file,'a') as f:
            f.write(mesg)
    
    # sum_EER = 0
    # total = 0
    EERs = []
    for e in range(hp.test.epochs):
        sum_EER, total = test_epoch(e, device, test_loader, embedder_net, reporter_net)
        EERs.append(sum_EER / total)
        # sum_EER += sub_ERR
        # total += subtotal
        # avg_EER = sum_EER / total
    EERs = torch.stack(EERs)
    # human-readable 
    avg_EER = EERs.mean()
    std_EER = EERs.std()
    avg_EER *= 100
    std_EER *= 100
    mesg = "\nAvg. EER across {0} epochs: {1:.2f}+-{2:.2f}".format(hp.test.epochs, avg_EER, std_EER)
    print(mesg)
    if hp.test.log_file is not None:
        with open(hp.test.log_file,'a') as f:
            f.write(mesg)
        
def train_epoch(e, device, train_loader, embedder_net, loss_net, reporter_net, optimizer, total_number):
    total_loss = 0
    for batch_id, (padded_Xs, length_Xs) in enumerate(train_loader): 
        padded_Xs = padded_Xs.to(device)
        length_Xs = length_Xs.to(device)
        
        #gradient accumulates
        optimizer.zero_grad()
        
        if hp.train.pretrain_epochs > 0:
            ret = embedder_net(padded_Xs, length_Xs, e > hp.train.pretrain_epochs)
        else:
            embedder_net.freeze()
            ret = embedder_net(padded_Xs, length_Xs)
        S_X = ret["S_X"]    # speaker embedding
        #get loss, call backward, step optimizer
        losses = {}
        reports = {}
        def update(embedding, suffix, loss=True):
            if loss:
              ret = loss_net(embedding)
              for k,v in ret.items():
                  if k in ret:
                      losses["{0}-{1}".format(k, suffix)] = v
                  else:
                      losses[k] = v
            if reporter_net is not None:
                ret = reporter_net(embedding)
                for k,v in ret.items():
                    if k in ret:
                        reports["{0}-{1}".format(k, suffix)] = v
                    else:
                        reports[k] = v
        if hp.train.M > 1:
            losses.update(loss_net(S_X))
            if reporter_net is not None:
                reports.update(reporter_net(S_X))
        length_X = ret["L"]
        batch_max_len = torch.max(length_X)
        N = length_X.size(0)
        M = length_X.size(1)
        filter_X = torch.zeros((N, M, batch_max_len, 1))
        for i in range(N):
            for j in range(M):
                filter_X[i, j, :length_X[i][j], 0] = torch.ones(length_X[i][j])
        filter_X = filter_X.to(device)
        # print(filter_X)
        if "X_hat" in ret:
            # losses["self"] =  F.mse_loss(ret["X"], ret["X_hat"])
            D_X = ret["X_hat"].size(-1)
            # print(ret["X_hat"])
            X_hat = ret["X_hat"] * filter_X
            loss = nn.MSELoss(reduce=False, reduction="none")
            losses["autovc-self"] = loss(X_hat, ret["X"]).sum() / filter_X.sum() / D_X
        if "X_tilda" in ret:
            # losses["self-pre"] =  F.mse_loss(ret["X"], ret["X_tilda"])
            D_X = ret["X_tilda"].size(-1)
            X_tilda = ret["X_tilda"] * filter_X
            loss = nn.MSELoss(reduce=False, reduction="none")
            losses["autovc-self-pre"] = loss(X_tilda, ret["X"]).sum() / filter_X.sum() / D_X
        if "C_X" in ret and "C_hat" in ret:
            # C_X and C_hat are masked out in the encoder
            # losses["content"] =  F.mse_loss(ret["C_X"], ret["C_hat"])
            D_C = ret["C_X"].size(-1)
            nonzero = length_X // hp.autovc.freq
            # C_* are not sparse
            # loss = nn.L1Loss(reduce=False, reduction="none")
            loss = nn.MSELoss(reduce=False, reduction="none")
            losses["content"] = loss(ret["C_X"], ret["C_hat"]).sum() / nonzero.sum() / D_C
        # mixed / generated loss (autovc-v2)
        if "S_hat" in ret:
            S_hat = ret["S_hat"]
            if hp.autovc.mix:
                S_mix = torch.cat((S_X, S_hat), dim=1)
                update(S_mix, "mix")
            elif hp.train.K > 1:
                update(S_hat, "gen")
        # triplet loss (autovc-v3)
        if "S_anc" in ret and "S_pos" in ret and "S_neg" in ret:
            triplet_loss = nn.TripletMarginLoss(margin=hp.autovc.margin, p=hp.train.norm)
            S_anc = ret["S_anc"]
            S_pos = ret["S_pos"]
            S_neg = ret["S_neg"]
            losses["style-diff"] = triplet_loss(S_anc, S_pos, S_neg)
            # just report
            S_mix = torch.cat((S_X, S_pos), dim=1)
            update(S_mix, "mix", loss=False)
        # triplet loss (autovc-v4)
        if "C_X_1" in ret and "C_V_2" in ret and "C_V_1" in ret and "C_V_2" in ret:
            C_anc = ret["C_V_1"]
            C_pos = ret["C_X_1"]
            C_neg = ret["C_V_2"]
            triplet_loss_nomargin = nn.TripletMarginLoss(margin=hp.autovc.margin)
            losses["content-diff"] = .5*triplet_loss_nomargin(C_anc, C_pos, C_neg)
            C_anc = ret["C_V_2"]
            C_pos = ret["C_X_2"]
            C_neg = ret["C_V_1"]
            triplet_loss_nomargin = nn.TripletMarginLoss(margin=hp.autovc.margin)
            losses["content-diff"] += .5*triplet_loss_nomargin(C_anc, C_pos, C_neg)
        # anti-collapse regularizer (autovc-v4)
        if "S_X_1" in ret and "S_V_2" in ret and "S_V_1" in ret and "S_V_2" in ret:
            p = hp.train.norm
            diff = ret["S_X_1"] - ret["S_X_2"]
            before = torch.norm(diff, p=p, dim=-1).pow(p)
            diff = ret["S_V_1"] - ret["S_V_2"]
            after = torch.norm(diff, p=p, dim=-1).pow(p)
            assert after.shape == before.shape
            losses["ba-ratio"] = torch.div(before, after).mean()
        # # triplet loss (autovc-v4)
        # # it seems to be non-sense
        # if "S_V_1" in ret and "S_V_2" in ret and "S_trg" in ret:
        #     S_V_1 = ret["S_V_1"]
        #     S_V_2 = ret["S_V_2"]
        #     S_trg = ret["S_trg"]
        #     triplet_loss_nomargin = nn.TripletMarginLoss(p=hp.train.norm)
        #     losses["style-same"] = triplet_loss_nomargin(S_V_1, S_V_2, S_trg)
            
        #TODO(hwidongna): weighted sum
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
        torch.nn.utils.clip_grad_norm_(loss_net.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss
        if hp.train.log_interval > 0 and (batch_id + 1) % hp.train.log_interval == 0:
            mesg = "{0}\tEpoch:{1}[{2}/{3}]\tLoss:{4:.4f}\tTLoss:{5:.4f}\t\n".format(time.ctime(), e+1,
                    batch_id+1, total_number//hp.train.N, loss, total_loss / (batch_id + 1))
            mesg += "losses"
            for k,v, in losses.items():
                mesg += ",\t{0}: {1:.4f}".format(k,v)
            mesg += "\n"
            mesg += "reports"
            for k,v, in reports.items():
                mesg += ",\t{0}: {1:.4f}".format(k,v)
            mesg += "\n"
            print(mesg)
            if hp.train.log_file is not None:
                with open(hp.train.log_file,'a') as f:
                    f.write(mesg)

def train():
    assert hp.train.N > 1
    assert hp.train.M > 0
    assert hp.train.K >= 0
    device = torch.device(hp.device)
    
    train_dataset = SpeakerDatasetSpectrogram(hp.train.M, hp.data.train_path)
    # https://stackoverflow.com/questions/55041080/how-does-pytorch-dataloader-handle-variable-size-data
    #TODO(hwidongna): https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N,
                              shuffle=True, collate_fn=custom_collate,
                              num_workers=hp.train.num_workers, drop_last=True) 
    
    test_dataset = SpeakerDatasetSpectrogram(hp.test.M, hp.data.test_path)
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N,
                              shuffle=True, collate_fn=custom_collate,
                              num_workers=hp.test.num_workers, drop_last=True) 
    print("Train using %s"%(hp.train.loss_type))
    reporter_net = Reporter(hp.train.reports, hp.train.norm)
    # default
    if not hp.train.loss_type.startswith("ge2e"):
        loss_net = LpDistanceLoss(device, hp.train.norm)
    if hp.train.loss_type.startswith("autovc-v"):
        V = hp.train.loss_type[len("autovc-v"):]
        K = hp.train.K
        embedder_net = eval("SpeechEmbedderAutoVCv{0}({1})".format(V, K)).to(device)
    elif hp.train.loss_type == "autovc":
        embedder_net = SpeechEmbedderAutoVC().to(device)
    elif hp.train.loss_type == "scaled-norm":
        embedder_net = SpeechEmbedderScaledNorm().to(device)
    elif hp.train.loss_type == "ge2e-contrast":
        embedder_net = SpeechEmbedder().to(device)
        loss_net = ContrastLoss(device)
        reporter_net = None
    elif hp.train.loss_type == "ge2e-softmax":
        embedder_net = SpeechEmbedder().to(device)
        loss_net = GE2ELoss(device)
        reporter_net = None
    else:
        raise Exception("unsupported loss type: %s"%hp.train.loss_type)

    if hp.train.optimizer.upper() == "SGD":
        opt = lambda params, **kwargs: torch.optim.SGD(params, **kwargs)
    elif hp.train.optimizer.upper() == "ADAM":
        opt = lambda params, **kwargs: torch.optim.Adam(params, **kwargs)
    else:
        raise Exception("unsupported optimizer: %s"%hp.train.optimizer)

    optimizer = opt([
                    {'params': embedder_net.parameters()},
                    {'params': loss_net.parameters()}
                ], lr=hp.train.lr)
    
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    # torch.autograd.set_detect_anomaly(True)
    
    start = 0
    if hp.train.restore:
        # embedder_net.load_state_dict(torch.load(hp.model.model_path))
        embedder_net.load_my_state_dict(torch.load(hp.model.model_path))
        start = int(re.findall(r"epoch_[\d]+", hp.model.model_path)[0][len("epoch_"):])
        print("Start from epoch %d"%start)
    embedder_net.eval()
    test_epoch(start, device, test_loader, embedder_net, reporter_net)
    embedder_net.train()
    for e in range(start, hp.train.epochs):
        if hp.train.checkpoint_interval > 0 and (e + 1) % hp.train.checkpoint_interval == 0:
            embedder_net.eval()
            test_epoch(e, device, test_loader, embedder_net, reporter_net)
            embedder_net.cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e+1) + ".pth"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()
        train_epoch(e, device, train_loader, embedder_net, loss_net, reporter_net, optimizer, len(train_dataset))
    #save model
    embedder_net.eval().cpu()
    save_model_filename = "final_epoch_" + str(e + 1) + ".model"
    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    torch.save(embedder_net.state_dict(), save_model_path)
    
    print("\nDone, trained model saved at", save_model_path)

if __name__=="__main__":
    if hp.training:
        train()
    else:
        test()
