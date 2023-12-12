## Audio Emotion Embedder: 
The solution pipeline goes as follows:
* Each WAV file is transformed into a Mel-spectrogram.
* Siamese network (ResNet18 backbone followed by fully connected layers) encodes each spectrogram into a vector.
* Triplet loss is used as an objective function, to that end, each example in the dataset (anchor) is assigned a random positive and negative examples.

####Run the data processing:
`python src/data/data_processing.py `

####Run the model training and evaluation:
`python src/models/train.py `