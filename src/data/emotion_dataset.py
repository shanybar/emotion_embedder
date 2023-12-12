
import torch
import random
import librosa
import torchaudio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchaudio.functional import amplitude_to_DB


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

    # def __getitem__(self, index):
    #     siganl_1_path, siganl_2_path, target = self._create_pair(index)
    #     signal_1, sr_1 = torchaudio.load(siganl_1_path)
    #     signal_2, sr_2 = torchaudio.load(siganl_2_path)
    #
    #     signal_1 = self._resample_if_necessary(signal_1, sr_1)
    #     signal_2 = self._resample_if_necessary(signal_2, sr_2)
    #
    #     signal_1 = self._mix_down_if_necessary(signal_1)
    #     signal_2 = self._mix_down_if_necessary(signal_2)
    #
    #     signal_1 = self._cut_if_necessary(signal_1)
    #     signal_2 = self._cut_if_necessary(signal_2)
    #
    #     signal_1 = self._right_pad_if_necessary(signal_1)
    #     signal_2 = self._right_pad_if_necessary(signal_2)
    #
    #     mel_signal_1 = self.mel_spectrogram(signal_1)
    #     mel_signal_2 = self.mel_spectrogram(signal_2)
    #
    #     mel_signal_1_db = torch.tensor(librosa.power_to_db(mel_signal_1))
    #     mel_signal_2_db = torch.tensor(librosa.power_to_db(mel_signal_2))
    #
    #     # mel_signal_1_db = amplitude_to_DB(mel_signal_1, 10.0, 0.0, 1.0, 80.0)
    #     # mel_signal_2_db = amplitude_to_DB(mel_signal_2, 10.0, 0.0, 1.0, 80.0)
    #
    #     return mel_signal_1_db, mel_signal_2_db, target

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

        anchor = torch.tensor(librosa.power_to_db(anchor))
        positive = torch.tensor(librosa.power_to_db(positive))
        negative = torch.tensor(librosa.power_to_db(negative))

        # mel_signal_1_db = amplitude_to_DB(mel_signal_1, 10.0, 0.0, 1.0, 80.0)
        # mel_signal_2_db = amplitude_to_DB(mel_signal_2, 10.0, 0.0, 1.0, 80.0)

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
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.max_len:
            signal = signal[:, :self.max_len]
        return signal

    def _right_pad_if_necessary(self, signal):
        signal_len = signal.shape[1]
        if signal_len < self.max_len:
            num_missing_samples = self.max_len - signal_len
            padding_last_dim = (0, num_missing_samples) # the form of(padding_left,padding_right)
            signal = torch.nn.functional.pad(signal, padding_last_dim)
        return signal

    def _group_examples(self):
        np_arr = self.targets

        # group examples based on class
        self.grouped_examples = {}
        for i in range(1, 9):
            self.grouped_examples[i] = np.where((np_arr == i))[0]

    def _create_pair(self, index):
        """
        """
        # pick some random class for the first image
        selected_class = random.randint(1, 8)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(1, self.grouped_examples[selected_class].shape[0] - 1)

        # pick the index to get the first image
        index_1 = self.grouped_examples[selected_class][random_index_1]

        siganl_1_path = self._get_audio_sample_path(index_1)
        selected_class = self._get_audio_sample_label(index_1)

        # same class
        if index % 2 == 0 and len(self.grouped_examples[selected_class]) >= 2:
            # pick a random index for the second image
            random_index_2 = random.randint(1, self.grouped_examples[selected_class].shape[0] - 1)
            index_2 = self.grouped_examples[selected_class][random_index_2]

            # ensure that the index of the second image isn't the same as the first image
            while index_2 == index_1:
                random_index_2 = random.randint(1, self.grouped_examples[selected_class].shape[0] - 1)
                index_2 = self.grouped_examples[selected_class][random_index_2]

            siganl_2_path = self._get_audio_sample_path(index_2)

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)

        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(1, 8)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class and len(self.grouped_examples[other_selected_class]) < 2:
                other_selected_class = random.randint(1, 8)

            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(1, self.grouped_examples[other_selected_class].shape[0] - 1)

            # pick the index to get the second image
            index_2 = self.grouped_examples[other_selected_class][random_index_2]

            # get the second image
            siganl_2_path = self._get_audio_sample_path(index_2)

            # set the label for this example to be negative (0)
            # target = torch.tensor(0, dtype=torch.float)
            target = torch.tensor(-1, dtype=torch.float)

        return siganl_1_path, siganl_2_path, target

    def _create_triplet(self, index):
        """
        """

        anchor_path = self._get_audio_sample_path(index)
        selected_class = self._get_audio_sample_label(index)

        # same class

        random_index_2 = random.randint(1, self.grouped_examples[selected_class].shape[0] - 1)
        index_2 = self.grouped_examples[selected_class][random_index_2]

        # ensure that the index of the second image isn't the same as the first image
        while index_2 == index:
            random_index_2 = random.randint(1, self.grouped_examples[selected_class].shape[0] - 1)
            index_2 = self.grouped_examples[selected_class][random_index_2]

        positive_path = self._get_audio_sample_path(index_2)

        # set the label for this example to be positive (1)
        # target = torch.tensor(1, dtype=torch.float)

        # different class

        # pick a random class
        other_selected_class = random.randint(1, 8)

        while other_selected_class == selected_class:
            other_selected_class = random.randint(1, 8)

        random_index_3 = random.randint(1, self.grouped_examples[other_selected_class].shape[0] - 1)

        index_3 = self.grouped_examples[other_selected_class][random_index_3]

        negative_path = self._get_audio_sample_path(index_3)

        # set the label for this example to be negative (0)
        # target = torch.tensor(0, dtype=torch.float)
        # target = torch.tensor(-1, dtype=torch.float)

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

if __name__ == "__main__":
    ANNOTATIONS_FILE = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\train_annotations.csv"

    dataset = EmotionDataset(ANNOTATIONS_FILE, target_sample_rate=22000, max_len=22000*4)
    print(f"There are {len(dataset)} samples in the dataset.")
    signal_1, signal_2, label = dataset[80]

    plot_spectrogram(torch.squeeze(signal_1))
    print(f"Label: {label}")

