#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Modified from https://github.com/HarryVolek/Speaker_Verification
import glob
import os
import librosa
import numpy as np
import random
from hparam import hparam as hp

# # downloaded dataset path
# audio_path = glob.glob(os.path.dirname(hp.unprocessed_data))                                        

# def save_spectrogram_tisv():
#     """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
#         Each partial utterance is splitted by voice detection using DB
#         and the first and the last 180 frames from each partial utterance are saved. 
#         Need : utterance data set (VTCK)
#     """
#     print("start text independent utterance feature extraction")
#     os.makedirs(hp.data.train_path, exist_ok=True)   # make folder to save train file
#     os.makedirs(hp.data.test_path, exist_ok=True)    # make folder to save test file

#     utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
#     total_speaker_num = len(audio_path)
#     train_speaker_num= (total_speaker_num//10)*9            # split total data 90% train and 10% test
#     print("total speaker number : %d"%total_speaker_num)
#     print("train : %d, test : %d"%(train_speaker_num, total_speaker_num-train_speaker_num))
#     for i, folder in enumerate(audio_path):
#         print("%dth speaker processing..."%i)
#         utterances_spec = []
#         for utter_name in os.listdir(folder):
#             if utter_name[-4:] == '.WAV':
#                 utter_path = os.path.join(folder, utter_name)         # path of each utterance
#                 utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
#                 intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection 
#                 # this works fine for timit but if you get array of shape 0 for any other audio change value of top_db
#                 # for vctk dataset use top_db=100
#                 for interval in intervals:
#                     if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
#                         utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
#                         S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
#                                               win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
#                         S = np.abs(S) ** 2
#                         mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
#                         S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
#                         utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
#                         utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance

#         utterances_spec = np.array(utterances_spec)
#         print(utterances_spec.shape)
#         if i<train_speaker_num:      # save spectrogram as numpy file
#             np.save(os.path.join(hp.data.train_path, "speaker%d.npy"%i), utterances_spec)
#         else:
#             np.save(os.path.join(hp.data.test_path, "speaker%d.npy"%(i-train_speaker_num)), utterances_spec)

# if __name__ == "__main__":
#     save_spectrogram_tisv()

def save_spectrogram_tisv(input_path, output_path, train=True):
    """ Full preprocess of text independent utterance. The log-mel-spectrogram is saved as numpy file.
        Each partial utterance is splitted by voice detection using DB
        and the first and the last 180 frames from each partial utterance are saved. 
        Need : utterance data set (VTCK)
    """
    if train:
      print("TRAIN:\nstart text independent utterance feature extraction")
    else:
      print("TEST:\nstart text independent utterance feature extraction")
    audio_path = glob.glob(os.path.dirname(input_path))
    os.makedirs(output_path, exist_ok=True)   # make folder to save file

    utter_min_len = (hp.data.tisv_frame * hp.data.hop + hp.data.window) * hp.data.sr    # lower bound of utterance length
    total_speaker_num = len(audio_path)
    low = hp.data.tisv_frame_min
    high = hp.data.tisv_frame_max
    print("total speaker number : %d"%total_speaker_num)
    for i, folder in enumerate(audio_path):
        print("%dth speaker processing..."%(i+1))
        utterances_spec = []
        for utter_name in os.listdir(folder):
            if utter_name[-4:] == '.WAV':
                utter_path = os.path.join(folder, utter_name)         # path of each utterance
                utter, sr = librosa.core.load(utter_path, hp.data.sr)        # load utterance audio
                intervals = librosa.effects.split(utter, top_db=30)         # voice activity detection 
                for interval in intervals:
                    if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                        utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                        S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                                              win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                        S = np.abs(S) ** 2
                        mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                        S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                        utterances_spec.append(S[:, :hp.data.tisv_frame])    # first 180 frames of partial utterance
                        utterances_spec.append(S[:, -hp.data.tisv_frame:])   # last 180 frames of partial utterance

                # for interval in intervals:
                #     if not train or (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
                #         utter_part = utter[interval[0]:interval[1]]         # save first and last 180 frames of spectrogram.
                #         S = librosa.core.stft(y=utter_part, n_fft=hp.data.nfft,
                #                               win_length=int(hp.data.window * sr), hop_length=int(hp.data.hop * sr))
                #         S = np.abs(S) ** 2
                #         mel_basis = librosa.filters.mel(sr=hp.data.sr, n_fft=hp.data.nfft, n_mels=hp.data.nmels)
                #         S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances
                
                #         # print("STFT %s"%(S))
                #         # print("\nSTFT shape [%d, %d]"%(S.shape[0], S.shape[1]))
                #         total_frames = int(S.shape[1])
                #         start = 0
                #         while start < total_frames - low:
                #             if train:
                #                 nframe = random.randint(low, high)
                #             else:
                #                 nframe = (low + high) // 2
                #             # print("nframe %s "%(nframe))
                #             end = min(start+nframe, total_frames)
                #             nframe = end-start
                #             # print("type %s "%(type(start)))
                #             # print("type %s "%(type(end)))
                #             # print("part [%d:%d)"%(start, end))
                #             part = S[:, start:end]
                #             # print("shape %s"%(str(part.shape)))
                #             # utterances_spec.append(part)
                #             padding = ((0,0), (0, hp.data.tisv_frame - nframe))
                #             padded = np.pad(part, padding)
                #             # print("padded shape %s"%(str(padded.shape)))
                #             utterances_spec.append(padded)
                #             if train:
                #                 start += nframe
                #             else:
                #                 start += (low + high) // 4

        utterances_spec = np.array(utterances_spec)
        print(utterances_spec.shape)
        np.save(os.path.join(output_path, "speaker%d.npy"%i), utterances_spec)

if __name__ == "__main__":
    save_spectrogram_tisv(hp.data.train_path_unprocessed, hp.data.train_path, True)
    save_spectrogram_tisv(hp.data.test_path_unprocessed, hp.data.test_path, False)
