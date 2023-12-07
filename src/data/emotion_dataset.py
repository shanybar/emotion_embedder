import os
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset


class EmotionDataset(Dataset):

    def __init__(self,
                 annotation_file,
                 audio_dir,
                 transformation,
                 target_sample_rate,
                 max_len,
                 device):
        self.annotations = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.max_len = max_len
        self.device = device
        self.TARGET_SAMPLE_RATE = 16000

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        audio_sample_path = self._get_audio_sample_path(idx)
        label = self._get_audio_sample_label(idx)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr, self.device)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)

        return signal, label


    def _get_audio_sample_path(self, idx):
        path = self.annotations.iloc[idx, 0]
        return path

    def _get_audio_sample_label(self, idx):
        return self.annotations.iloc[idx, 6]

    def _resample_if_necessary(self, signal, sr, device):
        if sr != self.TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, self.TARGET_SAMPLE_RATE)
            resampler = resampler.to(self.device)
            signal = resampler(signal)

        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        if signal[1] > self.max_len:
            signal = signal[:, :self.max_len]
        return signal

    def _right_pad_if_necessary(self, signal):
        signal_len = signal.shape[1]
        if signal_len < self.max_len:
            num_missing_samples = self.max_len - signal_len
            padding_last_dim = (0, num_missing_samples) # the form of(padding_left,padding_right)
            signal = torch.nn.functional.pad(signal, padding_last_dim)
        return signal


