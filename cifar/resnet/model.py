'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel1=3, kernel2=3, shortcut_kernel=1, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel1, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel2, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=shortcut_kernel, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, kernel1=1, kernel2=3, kernel3=1, shortcut_kernel=1, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel2, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=kernel3, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=shortcut_kernel, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, initial_kernel=3,
                 b1kernel1=3, b1kernel2=1, b1kernel3=3,
                 b2kernel1=3, b2kernel2=1, b2kernel3=3,
                 b3kernel1=3, b3kernel2=1, b3kernel3=3,
                 b4kernel1=3, b4kernel2=1, b4kernel3=3,
                 num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.initial_kernel = initial_kernel
        self.b1kernel1 = b1kernel1
        self.b1kernel2 = b1kernel2
        self.b1kernel3 = b1kernel3
        self.b2kernel1 = b2kernel1
        self.b2kernel2 = b2kernel2
        self.b2kernel3 = b2kernel3
        self.b3kernel1 = b3kernel1
        self.b3kernel2 = b3kernel2
        self.b3kernel3 = b3kernel3
        self.b4kernel1 = b4kernel1
        self.b4kernel2 = b4kernel2
        self.b4kernel3 = b4kernel3

        self.conv1 = nn.Conv2d(3, 64, kernel_size=self.initial_kernel, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, planes=64, num_blocks=num_blocks[0], kernel1=self.b1kernel1, kernel2=self.b1kernel2, kernel3=self.b1kernel3, stride=1)
        self.layer2 = self._make_layer(block, planes=128, num_blocks=num_blocks[1], kernel1=self.b2kernel1, kernel2=self.b2kernel2, kernel3=self.b2kernel3, stride=2)
        self.layer3 = self._make_layer(block, planes=256, num_blocks=num_blocks[2], kernel1=self.b3kernel1, kernel2=self.b3kernel2, kernel3=self.b2kernel3, stride=2)
        self.layer4 = self._make_layer(block, planes=512, num_blocks=num_blocks[3], kernel1=self.b4kernel1, kernel2=self.b4kernel2, kernel3=self.b2kernel3, stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, kernel1, kernel2, kernel3, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        print('block is {}'.format(block))
        #if block == 'BasicBlock':
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel1, kernel2, stride))
            self.in_planes = planes * block.expansion
        # else:
        #     for stride in strides:
        #         layers.append(block(self.in_planes, planes, kernel1, kernel2, kernel3, stride))
        #         self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet152()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

test()
