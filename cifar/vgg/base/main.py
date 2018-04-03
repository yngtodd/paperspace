'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from utils import get_train_valid_loader
from adaptive_model import VGG
#from model import VGG
from net import Net

import os
import time
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--use_cuda', type=bool, help="Whether to use cuda")
args = parser.parse_args()

if args.use_cuda:
    use_cuda = True
else:
    use_cuda = False


trainloader, validloader = get_train_valid_loader()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = VGG()
if use_cuda:
    net = nn.DataParallel(net)
    net.cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
optimizer = optim.Adam(net.parameters())


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        accuracy = 100.*correct/total
        if batch_idx % 10 == 0:
            print("Train loss: {:.4f}, Acc: {:.3f}".format(train_loss/(batch_idx+1), accuracy))
    print("\nTrain loss: {:.4f}, Acc: {:.3f}".format(train_loss, accuracy))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(validloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        accuracy = 100.*correct/total
    print("Test loss: {:.4f} Acc: {:.3f}".format(test_loss, accuracy))


def main():
    start_time = time.time()
    start_epoch=0
    for epoch in range(start_epoch, start_epoch+200):
        epoch_start = time.time()
        train(epoch)
        test(epoch)
        print('Epoch {:d} runtime: {:.4f}'.format(epoch, time.time()-epoch_start))
    print('Total runtime: {:.4f}'.format(time.time()-start_time))


if __name__ == '__main__':
    main()
