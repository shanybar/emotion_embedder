from __future__ import print_function

import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.data.emotion_dataset import EmotionDataset
from src.models.siamese_model import SiameseModel
from torch.optim.lr_scheduler import StepLR


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.TripletMarginLoss()

    for batch_idx, (mel_1, mel_2, mel_3) in enumerate(train_loader):
        mel_1, mel_2, mel_3 = mel_1.to(device), mel_2.to(device), mel_3.to(device)
        optimizer.zero_grad()
        output_1,output_2, output_3 = model(mel_1, mel_2, mel_3)
        loss = criterion(output_1,output_2, output_3)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(mel_1), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    criterion = nn.TripletMarginLoss()

    with torch.no_grad():
        for (mel_1, mel_2, mel_3) in test_loader:
            mel_1, mel_2, mel_3 = mel_1.to(device), mel_2.to(device), mel_3.to(device)
            output_1, output_2, output_3 = model(mel_1, mel_2, mel_3)
            loss = criterion(output_1, output_2, output_3)
            test_loss += loss

            dist_pos = F.cosine_similarity(output_1, output_2)
            dist_neg = F.cosine_similarity(output_1, output_3)

            res = torch.gt(dist_pos, dist_neg).sum().item()
            correct += res

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / (len(test_loader.dataset))))

    return test_loss


def train_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_csv_path = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\train_annotations.csv"
    val_csv_path = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\val_annotations.csv"

    train_dataset = EmotionDataset(train_csv_path, target_sample_rate=22000, max_len=22000*4)
    val_dataset = EmotionDataset(val_csv_path, target_sample_rate=22000, max_len=22000*4)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)

    model = SiameseModel().to(device)
    best_model = model
    best_loss = math.inf
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    for epoch in range(1, 15):
        train(1, model, device, train_loader, optimizer, epoch)
        val_loss = test(model, device, val_loader)
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)

    torch.save(best_model.state_dict(), "siamese_network.pt")

if __name__ == '__main__':
    train_model()