import os
import torch
import random
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class EmotionDataset(Dataset):

    def __init__(self,
                 annotation_file,
                 target_sample_rate,
                 max_len):
        label_mapping = {1: 'neutral', 2: 'calm'}
        # self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

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

    # def __getitem__(self, idx):
    #     audio_sample_path = self._get_audio_sample_path(idx)
    #     label = self._get_audio_sample_label(idx)
    #     signal, sr = torchaudio.load(audio_sample_path)
    #     signal = signal.to(self.device)
    #     signal = self._resample_if_necessary(signal, sr, self.device)
    #     signal = self._mix_down_if_necessary(signal)
    #     signal = self._cut_if_necessary(signal)
    #     signal = self._right_pad_if_necessary(signal)
    #     # signal = self.transformation(signal)
    #
    #     return signal, label

    def __getitem__(self, index):
        siganl_1_path, siganl_2_path, target = self._create_pair(index)
        signal_1, sr = torchaudio.load(siganl_1_path)
        signal_2, sr = torchaudio.load(siganl_2_path)
        signal_1 = self._resample_if_necessary(signal_1, sr)
        signal_2 = self._resample_if_necessary(signal_2, sr)

        signal_1 = self._cut_if_necessary(signal_1)
        signal_2 = self._cut_if_necessary(signal_2)

        signal_1 = self._right_pad_if_necessary(signal_1)
        signal_2 = self._right_pad_if_necessary(signal_2)

        mel_signal_1 = self.mel_spectrogram(signal_1)
        mel_signal_2 = self.mel_spectrogram(signal_2)

        return mel_signal_1, mel_signal_2, target

    def _get_audio_sample_path(self, idx):
        path = self.annotations.iloc[idx, 0]
        return path

    def _get_audio_sample_label(self, idx):
        return self.annotations.iloc[idx, 6]

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            # resampler = resampler.to(self.device)
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
        """
            To ease the accessibility of data based on the class, we will use `group_examples` to group
            examples based on class.

            Every key in `grouped_examples` corresponds to a class in the dataset.
             For every key in `grouped_examples`, every value will conform to all
              of the indices for the dataset examples that correspond to that key.
        """

        # np_arr = np.array(self.targets.clone())
        np_arr = self.targets

        # group examples based on class
        self.grouped_examples = {}
        for i in range(1, 9):
            self.grouped_examples[i] = np.where((np_arr == i))[0]

    def _create_pair(self, index):
        """
           For every example, we will select two images. There are two cases,
           positive and negative examples. For positive examples, we will have two
           images from the same class. For negative examples, we will have two images
           from different classes.
        """

        # pick some random class for the first image
        selected_class = random.randint(1, 8)

        # pick a random index for the first image in the grouped indices based of the label
        # of the class
        random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)

        # pick the index to get the first image
        index_1 = self.grouped_examples[selected_class][random_index_1]

        # get the first image
        # image_1 = self.data[index_1].clone().float()
        siganl_1_path = self._get_audio_sample_path(index_1)

        # same class
        if index % 2 == 0:
            # pick a random index for the second image
            random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)

            # ensure that the index of the second image isn't the same as the first image
            while random_index_2 == random_index_1:
                random_index_2 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)

            # pick the index to get the second image
            index_2 = self.grouped_examples[selected_class][random_index_2]

            # get the second image
            # image_2 = self.data[index_2].clone().float()
            siganl_2_path = self._get_audio_sample_path(index_2)

            # set the label for this example to be positive (1)
            target = torch.tensor(1, dtype=torch.float)

        # different class
        else:
            # pick a random class
            other_selected_class = random.randint(0, 9)

            # ensure that the class of the second image isn't the same as the first image
            while other_selected_class == selected_class:
                other_selected_class = random.randint(0, 9)

            # pick a random index for the second image in the grouped indices based of the label
            # of the class
            random_index_2 = random.randint(0, self.grouped_examples[other_selected_class].shape[0] - 1)

            # pick the index to get the second image
            index_2 = self.grouped_examples[other_selected_class][random_index_2]

            # get the second image
            # image_2 = self.data[index_2].clone().float()
            siganl_2_path = self._get_audio_sample_path(index_2)

            # set the label for this example to be negative (0)
            target = torch.tensor(0, dtype=torch.float)

        return siganl_1_path, siganl_2_path, target


if __name__ == "__main__":
    ANNOTATIONS_FILE = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\train_annotations.csv"

    dataset = EmotionDataset(ANNOTATIONS_FILE, target_sample_rate=16000, max_len=64000)
    print(f"There are {len(dataset)} samples in the dataset.")
    signal_1, signal_2, label = dataset[0]
    print(f"Label: {label}")

