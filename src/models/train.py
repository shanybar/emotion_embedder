from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse, random, copy
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
from src.data.emotion_dataset import EmotionDataset
from src.models.siamese_model import SiameseModel


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    for batch_idx, (images_1, images_2, targets) in enumerate(train_loader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            # if args.dry_run:
            #     break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    # we aren't using `TripletLoss` as the MNIST dataset is simple, so `BCELoss` can do the trick.
    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss


def train_model():
    # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch Siamese network Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=14, metavar='N',
    #                     help='number of epochs to train (default: 14)')
    # parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
    #                     help='learning rate (default: 1.0)')
    # parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
    #                     help='Learning rate step gamma (default: 0.7)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--no-mps', action='store_true', default=False,
    #                     help='disables macOS GPU training')
    # parser.add_argument('--dry-run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    # args = parser.parse_args()

    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_mps = not args.no_mps and torch.backends.mps.is_available()
    #
    # torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    #
    # if use_cuda:
    #     cuda_kwargs = {'num_workers': 1,
    #                    'pin_memory': True,
    #                    'shuffle': True}
    #     train_kwargs.update(cuda_kwargs)
    #     test_kwargs.update(cuda_kwargs)

    train_csv_path = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\train_annotations.csv"
    val_csv_path = "C:\\Users\\shany\\PycharmProjects\\emotion_embedder\\resources\\validation_annotations.csv"

    train_dataset = EmotionDataset(train_csv_path, target_sample_rate=16000, max_len=64000)
    val_dataset = EmotionDataset(val_csv_path, target_sample_rate=16000, max_len=64000)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = SiameseModel().to(device)
    best_model = model
    best_loss = math.inf
    optimizer = optim.Adadelta(model.parameters(), lr=0.0001)

    scheduler = StepLR(optimizer, step_size=1)
    for epoch in range(1, 10):
        train(10, model, device, train_loader, optimizer, epoch)
        val_loss = test(model, device, val_loader)
        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)

    # if args.save_model:
    torch.save(best_model.state_dict(), "siamese_network.pt")


if __name__ == '__main__':
    train_model()