import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence 
from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np
from hparam import hparam as hp

#Modified from https://github.com/auspicious3000/autovc

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class AdaIN(torch.nn.Module):
    def forward(self, x, y):
        """
        Normalize x using y
        """
        # x: (batch, seq, dim_x)
        # y: (batch, dim_y)
        N = x.size(0)
        assert N == y.size(0)
        mean_x = x.mean(dim=-1, keepdim=True)
        mean_y = y.mean(dim=-1).reshape(N,1,1)
        # std_x = x.std(dim=-1)
        std_x = torch.sum((x - mean_x)**2 + 1e-6, dim=-1, keepdim=True)
        std_y = y.std(dim=-1).reshape(N,1,1)
        return std_y * (x - mean_x) + mean_y

class Encoder(nn.Module):
    """Encoder module:
    """
    def __init__(self, dim_neck, freq):
        super(Encoder, self).__init__()
        n_mels = hp.data.nmels
        channel = hp.autovc.channel
        kernel = hp.autovc.kernel
        num_conv = hp.autovc.num_conv
        num_enc = hp.autovc.num_enc
        self.dim_neck = dim_neck
        self.freq = freq
        
        self.adain = AdaIN()

        convolutions = []
        for i in range(num_conv):
            conv_layer = nn.Sequential(
                ConvNorm(n_mels if i==0 else channel,
                         channel,
                         kernel_size=kernel, stride=1,
                         padding=kernel//2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(channel))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm = nn.LSTM(channel, dim_neck, num_enc, batch_first=True,
        bidirectional=True)

    def forward(self, x, length, c_org):
        # x.shape = (batch, seq, dim_x)
        # c_org.shape = (batch, dim_s)
        n_seq = x.size(1)
        x = x.squeeze(1).transpose(2,1)
        # c_org = c_org.unsqueeze(-1).expand(-1, -1, x.size(-1))
        # # x.shape = (batch, dim_x+dim_s, seq)
        # x = torch.cat((x, c_org), dim=1)
        x = self.adain(x, c_org)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        # x.shape = (batch, nseq, dim_h)
        
        self.lstm.flatten_parameters()

        packed_X = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm(packed_X)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        # out_{f,b}.shape = (batch, nseq, dim_h/2)
        out_f = outputs[:, :, :self.dim_neck]
        out_b = outputs[:, :, self.dim_neck:]
        
        codes = []
        for i in range(0, n_seq, self.freq):
            if i + self.freq < n_seq:
                codes.append(torch.cat((out_f[:,i+self.freq-1,:],out_b[:,i,:]), dim=-1))
            else:
                codes.append(torch.cat((out_f[:,n_seq-1,:],out_b[:,i,:]), dim=-1))
        return torch.cat([c.unsqueeze(1) for c in codes], dim=1)
        
class Decoder(nn.Module):
    """Decoder module:
    """
    def __init__(self, dim_neck, dim_pre):
        super(Decoder, self).__init__()
        n_mels = hp.data.nmels
        proj = hp.autovc.proj
        num_dec = hp.autovc.num_dec

        self.adain = AdaIN()
        
        self.lstm1 = nn.LSTM(dim_neck*2, dim_pre, 1, batch_first=True)
        
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(dim_pre,
                         dim_pre,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(dim_pre))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
        
        self.lstm2 = nn.LSTM(dim_pre, proj, num_dec, batch_first=True)
        
        self.linear_projection = LinearNorm(proj, n_mels)

    def forward(self, codes, length, c_trg):
        freq = hp.autovc.freq
        N, T, D = codes.shape
        code_exp = codes.unsqueeze(2).expand(-1,-1,freq,-1).reshape(N,T*freq,D)
        
        # [batch, seq, dim_c] -> [batch, seq, dim_c+dim_s]
        # x = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1,code_exp.size(1),-1)), dim=-1)
        x = code_exp
        x = self.adain(x, c_trg)
        self.lstm1.flatten_parameters()
        
        packed_X = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm1(packed_X)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x.transpose(1, 2)
        
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        
        self.lstm2.flatten_parameters()
       
        packed_X = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        outputs, _ = self.lstm2(packed_X)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        
        decoder_output = self.linear_projection(outputs)

        return decoder_output   
    
    
class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        n_mels = hp.data.nmels
        channel = hp.autovc.channel
        kernel = hp.autovc.kernel
        num_post = hp.autovc.num_post

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mels, channel,
                         kernel_size=kernel, stride=1,
                         padding=kernel//2,
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(channel))
        )

        for i in range(1, num_post - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(channel,
                             channel,
                             kernel_size=5, stride=1,
                             padding=2,
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(channel))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(channel, n_mels,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(n_mels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))

        x = self.convolutions[-1](x)

        return x    
    

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, dim_neck, dim_pre, freq):
        super(Generator, self).__init__()
        
        self.encoder = Encoder(dim_neck, freq)
        self.decoder = Decoder(dim_neck, dim_pre)
        self.postnet = Postnet()

    def encoder_fn(self, x, length, c_org):
        return self.encoder(x, length, c_org)

    def decoder_fn(self, codes, length, c_trg):
        # [batch, seq/fre, dim_c] -> [batch, seq, dim_c]
        mel_outputs = self.decoder(codes, length, c_trg)
                
        # [batch, seq, dim_x] -> [batch, seq, dim_x]
        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2,1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2,1)
        
        return mel_outputs, mel_outputs_postnet

    def forward(self, x, length, c_org, c_trg):
        """x: """
                
        # [batch, seq, dim_x] -> list of [batch, dim_c]
        codes = self.encoder_fn(x, length, c_org)
        if c_trg is None:
            return codes
        
        mel_outputs, mel_outputs_postnet = self.decoder_fn(codes, length, c_trg)
        return mel_outputs, mel_outputs_postnet, codes

    
