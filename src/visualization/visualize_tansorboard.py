import numpy as np
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from src.data.emotion_dataset import EmotionDataset
from src.models.siamese_model import SiameseModel

# init writer
output_path = 'C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\src\\visualization\\results'
os.makedirs(output_path, exist_ok=True)
writer = SummaryWriter(output_path)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# init model
model = SiameseModel().to(device)
model.load_state_dict(torch.load("C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\src\\models\\siamese_network_loss08.pt"))
model.eval()

# load dataset
val_csv_path = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\" \
               "val_annotations.csv"
val_dataset = EmotionDataset(val_csv_path, target_sample_rate=22000, max_len=22000*4)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

# get the class labels for each signal
class_labels = val_dataset.targets

# inference
features = []
with torch.no_grad():
    for (image_1, _, _) in val_loader:
        image_1 = image_1.to(device)
        output = model.forward_once(image_1).squeeze().detach().cpu().numpy()
        features.append(output)

# log embeddings
writer.add_embedding(np.array(features),
                     metadata=class_labels)
writer.close()
