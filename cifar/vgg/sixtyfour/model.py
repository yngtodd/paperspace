import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, kernel1=3, kernel2=3, kernel3=3, kernel4=3, kernel5=6,
                 kernel6=3, kernel7=3, kernel8=3, kernel9=3, kernel10=3,
                 dropout1=0.5, dropout2=0.5, dropout3=0.5, dropout4=0.5,
                 dropout5=0.5, dropout6=0.5):
        super(VGG, self).__init__()
        """
        Parameters:
        ----------
            * `input_size` (int)
                Input dimension for the data.
            * `kernel*` (int, defualt=3)
                Convolutional filter size for layer *.
            * `dropout* (int, default=0.5)`
                dropout at the end of block *.
        """
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3
        self.kernel4 = kernel4
        self.kernel5 = kernel5
        self.kernel6 = kernel6
        self.kernel7 = kernel7
        self.kernel8 = kernel8
        self.kernel9 = kernel9
        self.kernel10 = kernel10
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.dropout3 = dropout3
        self.dropout4 = dropout4
        self.dropout5 = dropout5
        self.dropout6 = dropout6

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=self.kernel1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, self.kernel2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            #nn.Dropout(self.dropout1),
            #nn.ReLU()
            )

        self.block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=self.kernel3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=self.kernel4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout2)
            #nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=self.kernel5, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=self.kernel6, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=self.kernel7),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout3)
            #nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=self.kernel8, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=self.kernel9, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=self.kernel10, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout4),
            #nn.ReLU()
            nn.AvgPool2d(kernel_size=1, stride=1)
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(512, 10) # 4096, 8192, 16384
            #nn.Dropout(self.dropout6),
            #nn.Linear(128, 10)
        )

    def forward(self, input):
#        print("input has shape {}".format(input.size()))
        x = self.block1(input)
#        print("output of block one has shape {}".format(x.size())) 
        x = self.block2(x)
#        print("Output of block two has shape {}".format(x.size()))
        x = self.block3(x)
#        print("Output of blick three has shape {}".format(x.size()))
        x = self.block4(x)
#        print("output of block four has shape {}".format(x.size()))
        x = x.view(x.size(0), -1)
#        print("viewing size has shape {}".format(x.size()))
        x = self.linear_layers(x)
        return x
