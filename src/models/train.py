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

    # criterion = nn.BCELoss()
    criterion = nn.CosineEmbeddingLoss(margin=0.0)
    criterion = nn.TripletMarginLoss()

    for batch_idx, (images_1, images_2, images_3) in enumerate(train_loader):
        images_1, images_2, images_3 = images_1.to(device), images_2.to(device), images_3.to(device)
        optimizer.zero_grad()
        # outputs = model(images_1, images_2).squeeze()
        # output_1,output_2 = model(images_1, images_2)
        output_1,output_2, output_3 = model(images_1, images_2, images_3)
        loss = criterion(output_1,output_2, output_3)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # criterion = nn.BCELoss()
    # criterion = nn.CosineEmbeddingLoss(margin=0.0)
    criterion = nn.TripletMarginLoss()

    with torch.no_grad():
        # for (images_1, images_2, targets) in test_loader:
        for (images_1, images_2, images_3) in test_loader:
            # images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            images_1, images_2, images_3 = images_1.to(device), images_2.to(device), images_3.to(device)
            # outputs = model(images_1, images_2).squeeze()
            # output_1, output_2 = model(images_1, images_2)
            # loss = criterion(output_1, output_2, targets).item()
            output_1, output_2, output_3 = model(images_1, images_2, images_3)
            loss = criterion(output_1, output_2, output_3)
            test_loss += loss
            # test_loss += criterion(outputs, targets.view_as(outputs)).sum().item()  # sum up batch loss
            # cos = F.l1(output_1, output_2)
            # pred = torch.where(cos > 0.7, 1, -1)  # get the index of the max log-probability
            # correct += pred.eq(targets.view_as(pred)).sum().item()

    # test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss


def train_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_csv_path = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\train_annotations.csv"
    val_csv_path = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\val_annotations.csv"
    # test_csv_path = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\test_annotations.csv"

    train_dataset = EmotionDataset(train_csv_path, target_sample_rate=22000, max_len=22000*4)
    val_dataset = EmotionDataset(val_csv_path, target_sample_rate=22000, max_len=22000*4)
    # test_dataset = EmotionDataset(test_csv_path, target_sample_rate=48000, max_len=48000*4)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SiameseModel().to(device)
    best_model = model
    best_loss = math.inf
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    for epoch in range(1, 20):
        train(1, model, device, train_loader, optimizer, epoch)
        val_loss = test(model, device, val_loader)
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)

    torch.save(best_model.state_dict(), "siamese_network.pt")

    # model = SiameseModel().to(device)
    # model.load_state_dict(torch.load("C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\src\\models\\siamese_network.pt"))
    # model.eval()
    # test_loss = test(model, device, test_loader)


if __name__ == '__main__':
    train_model()