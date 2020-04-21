#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 20:55:52 2018

@author: harry
"""
import glob
import numpy as np
import os
import random
from random import shuffle
import torch
from torch.utils.data import Dataset

from hparam import hparam as hp
from utils import mfccs_and_spec

class SpeakerDatasetTIMIT(Dataset):
    
    def __init__(self):

        if hp.training:
            self.path = hp.data.train_path_unprocessed
            self.utterance_number = hp.train.M
        else:
            self.path = hp.data.test_path_unprocessed
            self.utterance_number = hp.test.M
        self.speakers = glob.glob(os.path.dirname(self.path))
        
    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        
        speaker = self.speakers[idx]
        wav_files = glob.glob(speaker+'/*.WAV')
        shuffle(wav_files)
        wav_files = wav_files[0:self.utterance_number]
        
        mel_dbs = []
        for f in wav_files:
            _, mel_db, _ = mfccs_and_spec(f, wav_process = True)
            mel_dbs.append(mel_db)
        return torch.Tensor(mel_dbs)

class SpeakerDatasetTIMITPreprocessed(Dataset):
    
    def __init__(self, shuffle=True, utter_start=0):
        
        # data path
        if hp.training:
            self.path = hp.data.train_path
            self.utter_num = hp.train.M
        else:
            self.path = hp.data.test_path
            self.utter_num = hp.test.M
        self.file_list = os.listdir(self.path)
        self.shuffle=shuffle
        self.utter_start = utter_start
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        assert idx < len(self.file_list) 
        selected_file = self.file_list[idx]
        
        utters = np.load(os.path.join(self.path, selected_file))        # load utterance spectrogram of selected speaker
        if self.shuffle:
            utter_index = np.random.randint(0, utters.shape[0], self.utter_num)   # select M utterances per speaker
            utterance = utters[utter_index]       
        else:
            utterance = utters[self.utter_start: self.utter_start+self.utter_num] # utterances of a speaker [batch(M), n_mels, frames]

        utterance = utterance[:,:,:160]               # TODO implement variable length batch size

        utterance = torch.tensor(np.transpose(utterance, axes=(0,2,1)))     # transpose [batch, frames, n_mels]
        return utterance

class SpeakerDatasetSpectrogram(Dataset):
    
    def __init__(self, M, path, shuffle=True, utter_start=0, pad_token=0):
        
        # data path
        self.path = path
        self.utter_num = M
        self.spk_list = os.listdir(self.path)
        self.shuffle=shuffle
        self.utter_start = utter_start
        self.pad_token = pad_token
        
    def __len__(self):
        return len(self.spk_list)

    def __getitem__(self, idx):
        assert idx < len(self.spk_list) 
        selected_spk = self.spk_list[idx]
        spk_path = os.path.join(self.path, selected_spk)
        X = []
        for utter_path in os.listdir(spk_path):
            utter = np.load(os.path.join(spk_path, utter_path))        # load spectrogram of selected speaker
            X.append(np.transpose(utter, axes=(1,0)))

        assert len(X) > 0
        if self.shuffle:
            random.shuffle(X)

        X = X[self.utter_start: self.utter_start+self.utter_num] # select M per speaker
        return X
