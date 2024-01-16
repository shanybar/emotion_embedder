## Audio Emotion Embedder
This repository contains an implementation of a model that spans an emotion embedding space, using audio data from the
Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) dataset.

The training pipeline goes as follows:
* Each WAV file is transformed into a Mel-spectrogram.
* Siamese network (ResNet18 backbone followed by fully connected layers) encodes each spectrogram into a vector.
* Triplet loss is used as an objective function, to that end, each example in the dataset (anchor) is assigned a random positive (same class) and negative (another class) examples.

#### Run the data processing (EDA & splits the data into train & validation):
`python src/data/data_processing.py `

#### Run the model training and evaluation:
`python src/models/train.py `