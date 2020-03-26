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

from hparam import hparam as hp
from data_load import SpeakerDatasetTIMIT, SpeakerDatasetTIMITPreprocessed
from speech_embedder_net import SpeechEmbedder, EuclideanDistanceLoss
from utils import get_centroids, get_distance_forall, normalize_0_1
import torch.nn.functional as F

def train():
    assert hp.train.N > 1
    assert hp.train.M > 1
    device = torch.device(hp.device)
    
    if hp.data.data_preprocessed:
        train_dataset = SpeakerDatasetTIMITPreprocessed()
    else:
        train_dataset = SpeakerDatasetTIMIT()
    train_loader = DataLoader(train_dataset, batch_size=hp.train.N, shuffle=True, num_workers=hp.train.num_workers, drop_last=True) 
    
    embedder_net = SpeechEmbedder().to(device)
    if hp.train.restore:
        embedder_net.load_state_dict(torch.load(hp.model.model_path))
    loss_net = EuclideanDistanceLoss(device)
    opt = lambda params, **kwargs: torch.optim.SGD(params, **kwargs)
    #Both net and loss have trainable parameters
    # if hp.train.optimizer.lower() == "sgd":
    #     if hp.train.momentum is not None:
    #         opt = lambda params, **kwargs: torch.optim.SGD(params, momentum=hp.train.momentum, nesterov=True, **kwargs)
    #     else:
    #         opt = lambda params, **kwargs: torch.optim.SGD(params, **kwargs)
    # elif hp.train.optimizer.lower() == "adam":
    #     opt = lambda params, **kwargs: torch.optim.Adam(params, **kwargs)
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
        for batch_id, mel_db_batch in enumerate(train_loader): 
            mel_db_batch = mel_db_batch.to(device)
            
            mel_db_batch = torch.reshape(mel_db_batch, (hp.train.N*hp.train.M, mel_db_batch.size(2), mel_db_batch.size(3)))
            perm = random.sample(range(0, hp.train.N*hp.train.M), hp.train.N*hp.train.M)
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch[perm]
            #gradient accumulates
            optimizer.zero_grad()
            
            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (hp.train.N, hp.train.M, embeddings.size(1)))
            
            #get loss, call backward, step optimizer
            loss = loss_net(embeddings) #wants (Speaker, Utterances, embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(loss_net.parameters(), 1.0)
            optimizer.step()
            
            total_loss = total_loss + loss
            iteration += 1
            if (batch_id + 1) % hp.train.log_interval == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(), e+1,
                        batch_id+1, len(train_dataset)//hp.train.N, iteration,loss, total_loss / (batch_id + 1))
                print(mesg)
                if hp.train.log_file is not None:
                    with open(hp.train.log_file,'a') as f:
                        f.write(mesg)
                    
        if hp.train.checkpoint_dir is not None and (e + 1) % hp.train.checkpoint_interval == 0:
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

def test():
    device = torch.device(hp.device)
    
    if hp.data.data_preprocessed:
        test_dataset = SpeakerDatasetTIMITPreprocessed()
    else:
        test_dataset = SpeakerDatasetTIMIT()
    test_loader = DataLoader(test_dataset, batch_size=hp.test.N, shuffle=True, num_workers=hp.test.num_workers, drop_last=True)
    
    embedder_net = SpeechEmbedder().to(device)
    embedder_net.load_state_dict(torch.load(hp.model.model_path))
    embedder_net.eval()
    
    avg_EER = 0
    for e in range(hp.test.epochs):
        batch_avg_EER = 0
        for batch_id, mel_db_batch in enumerate(test_loader):
            mel_db_batch = mel_db_batch.to(device)
            #TODO(hwidong.na): k-shot test
            #TODO(hwidong.na): randomize enrollment / verification
            assert hp.test.M > hp.test.K
            perm = random.sample(range(0,hp.test.N*hp.test.M), hp.test.N*hp.test.M)
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            mel_db_batch = mel_db_batch.view(hp.test.N*hp.test.M, mel_db_batch.size(2), mel_db_batch.size(3))
            mel_db_batch = mel_db_batch[perm]
            embeddings = embedder_net(mel_db_batch)
            embeddings = embeddings[unperm]
            embeddings = torch.reshape(embeddings, (hp.test.N, hp.test.M, embeddings.size(1)))

            enrollment_embeddings = embeddings[:,:hp.test.K,:]
            enrollment_centroids = get_centroids(enrollment_embeddings)
            verification_embeddings = embeddings[:,hp.test.K:,:]
            # shape.d_matrix = (hp.test.K, hp.test.M-hp.test.K, hp.test.K)
            d_matrix = get_distance_forall(verification_embeddings, enrollment_centroids)
            # because the distance is not bounded, need to normalize 
            d_matrix = F.normalize(d_matrix, dim=2)
            # calculate ERR excluding enrollment
            
            MminusK = hp.test.M-hp.test.K
            # calculating EER
            diff = 1 + 1e-6; EER=1; EER_thresh = 1; EER_FAR=1; EER_FRR=1
            
            for thres in [0.01*i for i in range(100)]:
                sim_matrix_thresh = d_matrix<thres
                
                pred = lambda i: sim_matrix_thresh[i].float().sum()
                true = lambda i: sim_matrix_thresh[i,:,i].float().sum()
                false_positive = lambda i: pred(i) - true(i)
                FAR = (sum([false_positive(i) for i in range(int(hp.test.N))])
                /(hp.test.N-1.0)/(float(MminusK))/hp.test.N)
    
                true_negative = lambda i: MminusK - true(i)
                FRR = (sum([true_negative(i) for i in range(int(hp.test.N))])
                /(float(MminusK))/hp.test.N)
                
                # Save threshold when FAR = FRR (=EER)
                if diff> abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
            # human-readable 
            EER *= 100
            EER_FAR *= 100
            EER_FRR *= 100
            mesg = "\nEER : %0.2f (thres:%.2f, FAR:%.2f, FRR:%.2f)"%(EER,EER_thresh,EER_FAR,EER_FRR)
            print(mesg)
            if hp.test.log_file is not None:
                with open(hp.test.log_file,'a') as f:
                    f.write(mesg)
        avg_EER += batch_avg_EER/(batch_id+1)
    avg_EER = avg_EER / hp.test.epochs
    # human-readable 
    avg_EER *= 100
    mesg = "\n EER across {0} epochs: {1:.2f}".format(hp.test.epochs, avg_EER)
    print(mesg)
    if hp.test.log_file is not None:
        with open(hp.test.log_file,'a') as f:
            f.write(mesg)
        
if __name__=="__main__":
    if hp.training:
        train()
    else:
        test()
