import matplotlib

matplotlib.use('Agg')
import torch
import random
import gzip
import codecs
import logging
import numpy as np
import _pickle as cPickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.data.emotion_dataset import EmotionDataset
from src.models.siamese_model import SiameseModel
import argparse

from itertools import cycle, islice

random.seed(1)
np.random.seed(1)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    '''
    This script loads mentions representation components (full representation, context vector,
    and dependent mentions vector), projects the representations and plots each mention in
    a scatter plot (each representation component in a separate plot).
    '''

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = SiameseModel().to(device)
    model.load_state_dict(torch.load("C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\src\\models\\siamese_network.pt"))
    model.eval()

    val_csv_path = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\validation_annotations.csv"
    val_dataset = EmotionDataset(val_csv_path, target_sample_rate=16000, max_len=64000)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    vecs = []
    with torch.no_grad():
        for (image_1, _, _) in val_loader:
            image_1 = image_1.to(device)
            output = model.forward_once(image_1)
            vecs.append(output)

    vecs = np.array(vecs)
    logger.info('Computing TSNE')
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vecs)
    label_mapping = {1: 'neutral', 2: 'calm'}
    fig = matplotlib.pyplot.figure(figsize=(25, 25))

    colors = np.array(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00',"#000000"])

    colorset = [colors[i] for i in vocab_cluster_id]

    ax = plt.axes()
    ax.scatter(Y[:, 0], Y[:, 1], s=700, c=colorset)
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        ax.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=25)

    fig.savefig(out_file + '.pdf', format='pdf', dpi=2000, bbox_inches='tight')

    logger.info('Done!')


if __name__ == '__main__':
    main()