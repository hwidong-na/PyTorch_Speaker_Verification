#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:49:16 2018

@author: harry
"""

import os
import random
import time
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from data_load import SpeakerDatasetSpectrogram
from speech_embedder_net import SpeechEmbedder, GE2ELoss, ContrastLoss
from speech_embedder_net import SpeechEmbedderV2, EuclideanDistanceLoss
from speech_embedder_net import SpeechEmbedderV3, AutoVCLoss
from utils import get_centroids, get_cosdiff, get_distance_forall


def custom_collate(Xs):
    n_dim = hp.data.nmels
    # x.shape = (n_seq, n_dim)
    length_Xs = torch.tensor([tuple(x.shape[0] for x in X) for X in Xs])
    batch_max_len = torch.max(length_Xs)
    padded_Xs = torch.zeros((hp.train.N, hp.train.M, batch_max_len, n_dim))
    
    for i, X in enumerate(Xs):
        lens = length_Xs[i]
        for j, x in enumerate(X):
            padded_Xs[i, j, :lens[j]] = torch.tensor(x[:lens[j]])
    #TODO(hwidongna): sort by length?
    return padded_Xs, length_Xs

def train():
    assert hp.train.N > 1
    assert hp.train.M > 1
    device = torch.device(hp.device)
    
    train_dataset = SpeakerDatasetSpectrogram()
    # https://stackoverflow.com/questions/55041080/how-does-pytorch-dataloader-handle-variable-size-data
    #TODO(hwidongna): https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N,
                              shuffle=True, collate_fn=custom_collate,
                              num_workers=hp.train.num_workers, drop_last=True) 
    
    if hp.train.loss_type == "autovc":
        embedder_net = SpeechEmbedderV3().to(device)
        loss_net = AutoVCLoss(device)
    elif hp.train.loss_type == "euclidean":
        embedder_net = SpeechEmbedderV2().to(device)
        loss_net = EuclideanDistanceLoss(device)
    else:
        embedder_net = SpeechEmbedder().to(device)
        if hp.train.loss_type == "contrast":
            loss_net = ContrastLoss(device)
        else:
            loss_net = GE2ELoss(device)
    opt = lambda params, **kwargs: torch.optim.SGD(params, **kwargs)
    #Both net and loss have trainable parameters
    # for p in embedder_net.parameters():
    #     print(p)
    # for p in loss_net.parameters():
    #     print(p)
    optimizer = opt([
                    {'params': embedder_net.parameters()},
                    {'params': loss_net.parameters()}
                ], lr=hp.train.lr)
    
    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
    torch.autograd.set_detect_anomaly(True)
    
    embedder_net.train()
    iteration = 0
    for e in range(hp.train.epochs):
        total_loss = 0
        for batch_id, (padded_Xs, length_Xs) in enumerate(train_loader): 
            padded_Xs = padded_Xs.to(device)
            length_Xs = length_Xs.to(device)
            
            n_spk = padded_Xs.size(0)
            n_utt = padded_Xs.size(1)
            n_seq = padded_Xs.size(2)
            n_dim = padded_Xs.size(3)
            padded_X = padded_Xs.view(n_spk*n_utt, n_seq, n_dim)
            length_X = length_Xs.view(n_spk*n_utt)
            #gradient accumulates
            optimizer.zero_grad()
            
            embeddings = embedder_net((padded_X, length_X))
            embeddings = embeddings.view(n_spk, n_utt, embeddings.size(1))
            
            #get loss, call backward, step optimizer
            loss = loss_net(embeddings) #wants (Speaker, Utterances, embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(loss_net.parameters(), 1.0)
            optimizer.step()
            
            total_loss = total_loss + loss
            iteration += 1
            if hp.train.log_interval > 0 and (batch_id + 1) % hp.train.log_interval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(), e+1,
                        batch_id+1, len(train_dataset)//hp.train.N, iteration,loss, total_loss / (batch_id + 1))
                print(mesg)
                if hp.train.log_file is not None:
                    with open(hp.train.log_file,'a') as f:
                        f.write(mesg)
                    
        if hp.train.checkpoint_interval > 0 and (e + 1) % hp.train.checkpoint_interval == 0:
            embedder_net.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e+1) + ".pth"
            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()
    #save model
    embedder_net.eval().cpu()
    save_model_filename = "final_epoch_" + str(e + 1) + ".model"
    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
    torch.save(embedder_net.state_dict(), save_model_path)
    
    print("\nDone, trained model saved at", save_model_path)

#def train():
#    assert hp.train.N > 1
#    assert hp.train.M > 1
#    device = torch.device(hp.device)
    
#    if hp.data.data_preprocessed:
#        train_dataset = SpeakerDatasetTIMITPreprocessed()
#    else:
#        train_dataset = SpeakerDatasetTIMIT()
#    # when trainining, it is fine to drop last if shuffled
#    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True) 
    
#    if hp.train.restore:
#        embedder_net.load_state_dict(torch.load(hp.model.model_path))
#    if hp.train.loss_type == "euclidean":
#        embedder_net = SpeechEmbedderV2().to(device)
#        loss_net = EuclideanDistanceLoss(device)
#    else:
#        embedder_net = SpeechEmbedder().to(device)
#        if hp.train.loss_type == "contrast":
#            loss_net = ContrastLoss(device)
#        else:
#            loss_net = GE2ELoss(device)
#    opt = lambda params, **kwargs: torch.optim.SGD(params, **kwargs)
#    # if hp.train.optimizer.lower() == "sgd":
#    #     if hp.train.momentum is not None:
#    #         opt = lambda params, **kwargs: torch.optim.SGD(params, momentum=hp.train.momentum, nesterov=True, **kwargs)
#    #     else:
#    #         opt = lambda params, **kwargs: torch.optim.SGD(params, **kwargs)
#    # elif hp.train.optimizer.lower() == "adam":
#    #     opt = lambda params, **kwargs: torch.optim.Adam(params, **kwargs)
#    #Both net and loss have trainable parameters
#    optimizer = opt([
#                    {'params': embedder_net.parameters()},
#                    {'params': loss_net.parameters()}
#                ], lr=hp.train.lr)
    
#    os.makedirs(hp.train.checkpoint_dir, exist_ok=True)
#    torch.autograd.set_detect_anomaly(True)
    
#    embedder_net.train()
#    iteration = 0
#    for e in range(hp.train.epochs):
#        total_loss = 0
#        for batch_id, mel_db_batch in enumerate(train_loader): 
#            mel_db_batch = mel_db_batch.to(device)
            
#            n_spk = mel_db_batch.size(0)
#            n_utt = mel_db_batch.size(1)
#            n_seq = mel_db_batch.size(2)
#            n_dim = mel_db_batch.size(3)
#            mel_db_batch = mel_db_batch.view(n_spk*n_utt, n_seq, n_dim)
#            #gradient accumulates
#            optimizer.zero_grad()
            
#            embeddings = embedder_net(mel_db_batch)
#            embeddings = embeddings.view(n_spk, n_utt, embeddings.size(1))
            
#            #get loss, call backward, step optimizer
#            loss = loss_net(embeddings) #wants (Speaker, Utterances, embedding)
#            loss.backward()
#            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
#            torch.nn.utils.clip_grad_norm_(loss_net.parameters(), 1.0)
#            optimizer.step()
            
#            total_loss = total_loss + loss
#            iteration += 1
#            if hp.train.log_interval > 0 and (batch_id + 1) % hp.train.log_interval == 0:
#                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(), e+1,
#                        batch_id+1, len(train_dataset)//hp.train.N, iteration,loss, total_loss / (batch_id + 1))
#                print(mesg)
#                if hp.train.log_file is not None:
#                    with open(hp.train.log_file,'a') as f:
#                        f.write(mesg)
                    
#        if hp.train.checkpoint_interval > 0 and (e + 1) % hp.train.checkpoint_interval == 0:
#            embedder_net.eval().cpu()
#            ckpt_model_filename = "ckpt_epoch_" + str(e+1) + ".pth"
#            ckpt_model_path = os.path.join(hp.train.checkpoint_dir, ckpt_model_filename)
#            torch.save(embedder_net.state_dict(), ckpt_model_path)
#            embedder_net.to(device).train()
#    #save model
#    embedder_net.eval().cpu()
#    save_model_filename = "final_epoch_" + str(e + 1) + ".model"
#    save_model_path = os.path.join(hp.train.checkpoint_dir, save_model_filename)
#    torch.save(embedder_net.state_dict(), save_model_path)
    
#    print("\nDone, trained model saved at", save_model_path)

def test():
    device = torch.device(hp.device)
    
    if hp.data.data_preprocessed:
        test_dataset = SpeakerDatasetTIMITPreprocessed()
    else:
        test_dataset = SpeakerDatasetTIMIT()
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    
    if hp.train.loss_type == "euclidean":
        embedder_net = SpeechEmbedderV2().to(device)
    else:
        embedder_net = SpeechEmbedder().to(device)
    embedder_net.load_state_dict(torch.load(hp.model.model_path))
    embedder_net.eval()
    mesg = "Compute EER using %s"%(hp.train.loss_type)
    print(mesg)
    if hp.test.log_file is not None:
        with open(hp.test.log_file,'a') as f:
            f.write(mesg)
    
    sum_EER = 0
    total = 0
    for e in range(hp.test.epochs):
        for batch_id, mel_db_batch in enumerate(test_loader):
            mel_db_batch = mel_db_batch.to(device)
            # N.B. utterances are shuffled in SpeakerDatasetTIMIT*
            # k-shot test, k is the number enrollment utterences
            assert hp.test.M > hp.test.K
            n_spk = mel_db_batch.size(0)
            n_utt = mel_db_batch.size(1)
            n_seq = mel_db_batch.size(2)
            n_dim = mel_db_batch.size(3)

            mel_db_batch = mel_db_batch.view(n_spk*n_utt, n_seq, n_dim)
            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings.view(n_spk, n_utt, embeddings.size(1))

            enrollment_embeddings = embeddings[:,:hp.test.K,:]
            enrollment_centroids = get_centroids(enrollment_embeddings)
            verification_embeddings = embeddings[:,hp.test.K:,:]
            if hp.train.loss_type == "euclidean":
                d_matrix = get_distance_forall(verification_embeddings, enrollment_centroids)
                # because the distance is not bounded, need to normalize
                d_matrix = F.normalize(d_matrix, dim=2)
                threshold_fn = lambda thres: d_matrix < thres
            else:
                sim_matrix = get_cosdiff(verification_embeddings, enrollment_centroids)
                threshold_fn = lambda thres: sim_matrix > thres
            # calculate ERR excluding enrollment
            
            MminusK = hp.test.M-hp.test.K
            # calculating EER
            diff = 1 + 1e-6; EER=1; EER_thresh = 1; EER_FAR=1; EER_FRR=1
            
            for thres in [0.01*i for i in range(100)]:
                matrix_thresh = threshold_fn(thres)
                
                pred = lambda i: matrix_thresh[i].float().sum()
                true = lambda i: matrix_thresh[i,:,i].float().sum()
                false_acceptance = lambda i: pred(i) - true(i)
                FAR = (sum([false_acceptance(i) for i in range(int(n_spk))])
                /(n_spk-1.0)/(float(MminusK))/n_spk)
    
                false_rejection = lambda i: MminusK - true(i)
                FRR = (sum([false_rejection(i) for i in range(int(n_spk))])
                /(float(MminusK))/n_spk)
                
                # Save threshold when FAR = FRR (=EER)
                if diff> abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            sum_EER += EER*n_spk # need to weigh according to n_spk
            total += n_spk
            # human-readable 
            EER *= 100
            EER_FAR *= 100
            EER_FRR *= 100
            mesg = "\n%d-way EER : %.2f (thres:%.2f, FAR:%.2f, FRR:%.2f)"%(n_spk, EER,EER_thresh,EER_FAR,EER_FRR)
            print(mesg)
            if hp.test.log_file is not None:
                with open(hp.test.log_file,'a') as f:
                    f.write(mesg)
    avg_EER = sum_EER / total
    # human-readable 
    avg_EER *= 100
    mesg = "\nAvg. EER across {0} epochs: {1:.2f}".format(hp.test.epochs, avg_EER)
    print(mesg)
    if hp.test.log_file is not None:
        with open(hp.test.log_file,'a') as f:
            f.write(mesg)
        
if __name__=="__main__":
    if hp.training:
        train()
    else:
        test()
