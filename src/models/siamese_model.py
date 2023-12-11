import torch
import torchvision
from torchvision.models import ResNet18_Weights
import torch.nn as nn


class SiameseModel(nn.Module):
    """

    """

    def __init__(self):
        super(SiameseModel, self).__init__()

        self.resnet = torchvision.models.resnet18(weights = ResNet18_Weights.DEFAULT)

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features

        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to comp

        # self.fc = nn.Sequential(
        #     nn.Linear(self.fc_in_features * 2, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 1),
        # )

        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
        )

        self.sigmoid = nn.Sigmoid() #???

        # initialize the weights
        # self.resnet.apply(self.init_weights)
        # self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # get two signals' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


    # def forward(self, input1, input2):
    #     # get two signals' features
    #     output1 = self.forward_once(input1)
    #     output2 = self.forward_once(input2)
    #
    #     # concatenate both signals' features
    #     output = torch.cat((output1, output2), 1) #???
    #
    #     # pass the concatenation to the linear layers
    #     output = self.fc(output)
    #
    #     # pass the out of the linear layers to sigmoid layer
    #     output = self.sigmoid(output)
    #
    #     return output



