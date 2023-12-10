import os
import torch
import random
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
        # signal = self.transformation(signal)

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

    def group_examples(self):
        """
            To ease the accessibility of data based on the class, we will use `group_examples` to group
            examples based on class.

            Every key in `grouped_examples` corresponds to a class in MNIST dataset. For every key in
            `grouped_examples`, every value will conform to all of the indices for the MNIST
            dataset examples that correspond to that key.
        """

        np_arr = np.array(self.dataset.targets.clone())

        # group examples based on class
        self.grouped_examples = {}
        for i in range(0, 10):
            self.grouped_examples[i] = np.where((np_arr == i))[0]

    class APP_MATCHER(Dataset):
        def __init__(self, root, train, download=False):
            super(APP_MATCHER, self).__init__()

            # get MNIST dataset
            self.dataset = datasets.MNIST(root, train=train, download=download)

            # as `self.dataset.data`'s shape is (Nx28x28), where N is the number of
            # examples in MNIST dataset, a single example has the dimensions of
            # (28x28) for (WxH), where W and H are the width and the height of the image.
            # However, every example should have (CxWxH) dimensions where C is the number
            # of channels to be passed to the network. As MNIST contains gray-scale images,
            # we add an additional dimension to corresponds to the number of channels.
            self.data = self.dataset.data.unsqueeze(1).clone()

            self.group_examples()

        def group_examples(self):
            """
                To ease the accessibility of data based on the class, we will use `group_examples` to group
                examples based on class.

                Every key in `grouped_examples` corresponds to a class in MNIST dataset. For every key in
                `grouped_examples`, every value will conform to all of the indices for the MNIST
                dataset examples that correspond to that key.
            """

            # get the targets from MNIST dataset
            np_arr = np.array(self.dataset.targets.clone())

            # group examples based on class
            self.grouped_examples = {}
            for i in range(0, 10):
                self.grouped_examples[i] = np.where((np_arr == i))[0]

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, index):
            """
                For every example, we will select two images. There are two cases,
                positive and negative examples. For positive examples, we will have two
                images from the same class. For negative examples, we will have two images
                from different classes.

                Given an index, if the index is even, we will pick the second image from the same class,
                but it won't be the same image we chose for the first class. This is used to ensure the positive
                example isn't trivial as the network would easily distinguish the similarity between same images. However,
                if the network were given two different images from the same class, the network will need to learn
                the similarity between two different images representing the same class. If the index is odd, we will
                pick the second image from a different class than the first image.
            """

            # pick some random class for the first image
            selected_class = random.randint(0, 9)

            # pick a random index for the first image in the grouped indices based of the label
            # of the class
            random_index_1 = random.randint(0, self.grouped_examples[selected_class].shape[0] - 1)

            # pick the index to get the first image
            index_1 = self.grouped_examples[selected_class][random_index_1]

            # get the first image
            image_1 = self.data[index_1].clone().float()

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
                image_2 = self.data[index_2].clone().float()

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
                image_2 = self.data[index_2].clone().float()

                # set the label for this example to be negative (0)
                target = torch.tensor(0, dtype=torch.float)

            return image_1, image_2, target


