#!/usr/bin/env python3
from __future__ import print_function
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from torch import nn
import os
import sys

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATASET_ROOT_DIR = os.path.expanduser("./data")
NORMALIZATION_MEAN = [0.5, 0.5, 0.5]
NORMALIZATION_STD = [0.5, 0.5, 0.5]
EPOCHS = 5
TRAIN_BATCH = 4
VALIDATION_BATCH = 5

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _batch(mode):
    return TRAIN_BATCH if mode == 'train' else VALIDATION_BATCH


def _modes():
    return ['train', 'validation']


def preprocess_cifar10():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
    ])
    return transforms


class TemplateNet(nn.Module):
    def __init__(self, padding=False, num_convs=1):
        super(TemplateNet, self).__init__()
        if num_convs not in [1, 2]:
            raise ValueError('Maximum 2 convs')
        self.num_convs = num_convs

        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=128,
                               kernel_size=3,
                               padding=1 if padding else 0)  ## (A)
        self.conv2 = nn.Conv2d(in_channels=128,
                               out_channels=128,
                               kernel_size=3)  ## (B)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pre_fc1_size = 28800
        if padding and num_convs == 1:
            self.pre_fc1_size = 16 * 16 * 128
        elif padding and num_convs == 2:
            self.pre_fc1_size = 7 * 7 * 128
        elif not padding and num_convs == 2:
            self.pre_fc1_size = 6 * 6 * 128

        self.fc1 = nn.Linear(in_features=self.pre_fc1_size,
                             out_features=1000)  ## (C)
        self.fc2 = nn.Linear(in_features=1000, out_features=10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        if self.num_convs == 2:
            x = self.pool(F.relu(self.conv2(x)))  ## (D)
        x = x.view(-1, self.pre_fc1_size)  ## (E)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(net, loaders):
    net = net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(loaders['train']):
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                if i == 12000 - 1 and epoch == 1 - 1:
                    print("[epoch:%d, batch:%5d] loss: %.3f" %
                          (epoch + 1, i + 1, running_loss / float(2000)))
                    return
                running_loss = 0.0


def _question_1(loaders):
    net = TemplateNet()
    train(net, loaders)


def _question_2(loaders):
    net = TemplateNet(num_convs=2)
    train(net, loaders)


def _question_3(loaders):
    net = TemplateNet(num_convs=2, padding=True)
    train(net, loaders)
    return net


def _question_4(net, loaders):
    num_classes = len(loaders['validation'].dataset.classes)
    conf_matrx = torch.zeros(size=(num_classes, num_classes))
    net = net.to(DEVICE)
    for i, (x, y) in enumerate(loaders['validation']):
        x = x.to(DEVICE)
        y_pred = torch.argmax(net(x), dim=-1, keepdims=False)
        y_pred_cpu = y_pred.to(torch.device("cpu"))
        conf_matrx[y, y_pred_cpu] += 1

    print(conf_matrx)


def main(argv):

    loaders = {}
    for mode in _modes():
        dataset = torchvision.datasets.CIFAR10(root=DATASET_ROOT_DIR,
                                               train=mode == 'train',
                                               transform=preprocess_cifar10(),
                                               download=True)
        loaders[mode] = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=_batch(mode),
                                                    drop_last=True)
    _question_1(loaders)
    _question_2(loaders)
    net = _question_3(loaders)
    _question_4(net=net, loaders=loaders)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
