
import torch
import random
import librosa
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class EmotionDataset(Dataset):

    def __init__(self,
                 annotation_file,
                 target_sample_rate,
                 max_len):

        self.annotations = pd.read_csv(annotation_file)
        self.targets = self.annotations["label"].to_numpy()
        self._group_examples()
        self.target_sample_rate = target_sample_rate
        self.max_len = max_len

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
         sample_rate=self.target_sample_rate,
         n_fft=1024,
         hop_length=512,
         n_mels=64)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        anchor_path, positive_path, negative_path = self._create_triplet(index)
        anchor, sr_1 = torchaudio.load(anchor_path)
        positive, sr_2 = torchaudio.load(positive_path)
        negative, sr_3 = torchaudio.load(negative_path)

        anchor = self._resample_if_necessary(anchor, sr_1)
        positive = self._resample_if_necessary(positive, sr_2)
        negative = self._resample_if_necessary(negative, sr_3)

        anchor = self._mix_down_if_necessary(anchor)
        positive = self._mix_down_if_necessary(positive)
        negative = self._mix_down_if_necessary(negative)

        anchor = self._cut_if_necessary(anchor)
        positive = self._cut_if_necessary(positive)
        negative = self._cut_if_necessary(negative)

        anchor = self._right_pad_if_necessary(anchor)
        positive = self._right_pad_if_necessary(positive)
        negative = self._right_pad_if_necessary(negative)

        anchor = self.mel_spectrogram(anchor)
        positive = self.mel_spectrogram(positive)
        negative = self.mel_spectrogram(negative)

        # Used power_to_db to increase the signal
        anchor = torch.tensor(librosa.power_to_db(anchor))
        positive = torch.tensor(librosa.power_to_db(positive))
        negative = torch.tensor(librosa.power_to_db(negative))

        return anchor, positive, negative

    def _get_audio_sample_path(self, idx):
        path = self.annotations.iloc[idx, 0]
        return path

    def _get_audio_sample_label(self, idx):
        return self.annotations.iloc[idx, 1]

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)

        return signal

    def _mix_down_if_necessary(self, signal):
        """
        Reduce the number of channels if needed
        """
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        """
        Cut to self.max_len if needed
        """
        if signal.shape[1] > self.max_len:
            signal = signal[:, :self.max_len]
        return signal

    def _right_pad_if_necessary(self, signal):
        """
        Pad to self.max_len if needed
        """
        signal_len = signal.shape[1]
        if signal_len < self.max_len:
            num_missing_samples = self.max_len - signal_len
            padding_last_dim = (0, num_missing_samples) # the form of(padding_left,padding_right)
            signal = torch.nn.functional.pad(signal, padding_last_dim)
        return signal

    def _group_examples(self):
        """
        Group examples based on class
        """
        np_arr = self.targets

        self.grouped_examples = {}
        for i in range(1, 9):
            self.grouped_examples[i] = np.where((np_arr == i))[0]

    def _create_triplet(self, index):
        """
        Create a triplet of anchor, positive and negative examples for training with triplet loss
        """

        anchor_path = self._get_audio_sample_path(index)
        selected_class = self._get_audio_sample_label(index)

        # same class
        random_index_2 = random.randint(1, self.grouped_examples[selected_class].shape[0] - 1)
        index_2 = self.grouped_examples[selected_class][random_index_2]

        while index_2 == index:
            random_index_2 = random.randint(1, self.grouped_examples[selected_class].shape[0] - 1)
            index_2 = self.grouped_examples[selected_class][random_index_2]

        positive_path = self._get_audio_sample_path(index_2)

        # different class
        other_selected_class = random.randint(1, 8)

        while other_selected_class == selected_class:
            other_selected_class = random.randint(1, 8)

        random_index_3 = random.randint(1, self.grouped_examples[other_selected_class].shape[0] - 1)

        index_3 = self.grouped_examples[other_selected_class][random_index_3]

        negative_path = self._get_audio_sample_path(index_3)

        return anchor_path, positive_path, negative_path


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    spec_in_db = librosa.power_to_db(specgram)
    ax.imshow(spec_in_db, origin="lower", aspect="auto", interpolation="nearest")


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)

# if __name__ == "__main__":
#     ANNOTATIONS_FILE = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\train_annotations.csv"
#
#     dataset = EmotionDataset(ANNOTATIONS_FILE, target_sample_rate=22000, max_len=22000*4)
#     print(f"There are {len(dataset)} samples in the dataset.")
#     signal_1, signal_2, signal_3 = dataset[80]
#
#     plot_spectrogram(torch.squeeze(signal_1))


