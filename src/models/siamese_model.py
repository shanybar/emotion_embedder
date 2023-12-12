import torch
import torchvision
from torchvision.models import ResNet18_Weights
import torch.nn as nn


class SiameseModel(nn.Module):
    def __init__(self):
        super(SiameseModel, self).__init__()

        self.resnet = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # adjust to 1 channel input (not 2 channels like RGB)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2, input3):
        # get 3 signals' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        output3 = self.forward_once(input3)

        return output1, output2, output3



