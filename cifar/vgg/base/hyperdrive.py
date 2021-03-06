'''Distributed Hyperparameter Optimization of VGG over CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from utils import get_train_valid_loader
from model import VGG

from hyperspace import hyperdrive


parser = argparse.ArgumentParser(description='Setup experiment.')
parser.add_argument('--results_dir', type=str, help='Path to results directory.')
parser.add_argument('--use_cuda', type=bool, default=True, help='Whether to use cuda.')
parser.add_argument('--deadline', type=int, default=86400, help='Deadline (seconds) to finish within.')
args = parser.parse_args()

if args.use_cuda == True & torch.cuda.is_available() == False:
    print("Cuda not available, using CPU!")
    use_cuda = False
else:
    use_cuda = True

trainloader, validloader = get_train_valid_loader()
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def objective(params):
    kernel1 = int(params[0])
    kernel2 = int(params[1])
    kernel3 = int(params[2])
    kernel4 = int(params[3])
    kernel5 = int(params[4])
    kernel6 = int(params[5])
    kernel7 = int(params[6])
    kernel8 = int(params[7])
    kernel9 = int(params[8])
    kernel10 = int(params[9])
    dropout5 = float(params[10])
    dropout6 = float(params[11])

    net = VGG(kernel1=kernel1, kernel2=kernel2, kernel3=kernel3, kernel4=kernel4,
              kernel5=kernel5, kernel6=kernel6, kernel7=kernel7, kernel8=kernel8,
              kernel9=kernel9, kernel10=kernel10, dropout5=dropout5, dropout6=dropout6)

    if use_cuda and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
        net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    num_epochs = 50
    for _ in range(num_epochs):
        # Training
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
        #print("Train loss: {}".format(train_loss))

    # Validation
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(validloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    #print("Validation loss: {}".format(val_loss))

    return val_loss


def main():
    hparams = [(2, 10),       # kernel1
               (2, 10),       # kernel2
               (2, 10),       # kernel3
               (2, 10),       # kernel4
               (2, 10),       # kernel5
               (2, 10),       # kernel6
               (2, 10),       # kernel7
               (2, 10),       # kernel8
               (2, 10),       # kernel9
               (2, 10),       # kernel10
               (0.25, 0.95),  # dropout5
               (0.25, 0.95)]  # dropout6


    hyperdrive(objective=objective,
               hyperparameters=hparams,
               results_path=args.results_dir,
               model="GP",
               n_iterations=50,
               verbose=True,
               random_state=0,
               deadline=args.deadline)


if __name__ == '__main__':
    main()
