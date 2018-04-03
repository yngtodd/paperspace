import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, kernel1=3, kernel2=3):
        super(Net, self).__init__()
        """
        Parameters:
        ----------
            * `kernel*` (int, defualt=3)
                Convolutional filter size for layer *.
        """
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.conv1 = nn.Conv2d(3, 6, kernel_size=self.kernel1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=self.kernel2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        print("input size: {}".format(x.size()))
        x = self.pool(F.relu(self.conv1(x)))
        print("after first conv: {}".format(x.size()))
        x = self.pool(F.relu(self.conv2(x)))
        print("after second conv: {}".format(x.size()))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
