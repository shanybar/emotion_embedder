import torch
import random
import logging
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from src.data.emotion_dataset import EmotionDataset
from src.models.siamese_model import SiameseModel

random.seed(1)
np.random.seed(1)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = SiameseModel().to(device)
    model.load_state_dict(torch.load("C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\src\\models\\siamese_network_loss08.pt"))
    model.eval()

    val_csv_path = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\train_annotations.csv"
    val_dataset = EmotionDataset(val_csv_path, target_sample_rate=22000, max_len=22000*4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    labels = val_dataset.targets
    vecs = []
    with torch.no_grad():
        for (image_1, _, _) in val_loader:
            image_1 = image_1.to(device)
            output = model.forward_once(image_1).squeeze().detach().cpu().numpy()
            vecs.append(output)

    vecs = np.array(vecs)
    logger.info('Computing TSNE')
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vecs)
    label_mapping = {1: 'neutral', 2: 'calm', 3: 'happy' }
    fig = matplotlib.pyplot.figure(figsize=(25, 25))

    colors = np.array(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00',"#000000"])

    colorset = [colors[i] for i in labels]

    ax = plt.axes()
    ax.scatter(Y[:, 0], Y[:, 1], s=700, c=colorset)
    for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
        ax.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=25)

    fig.savefig('train_vis_val.pdf', format='pdf', dpi=2000, bbox_inches='tight')

    logger.info('Done!')


if __name__ == '__main__':
    main()