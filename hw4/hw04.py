#!/usr/bin/env python3
# Noureldin Hendy
# 00302449914

# All residual blocks variants are named `ResyBlockV*` (I called it Resy because it's res like :D)
#
# The ouput.txt are results of ResyBlockV3

# I mainly experimented with the projection and the fusion.
# Projection:
# On the projection, motivated by the thought
# of offloading the learning to the network, I made the projection
# layer a 3x3 conv layer with number of channels equal to number of
# output channels (ResyBlockV1) and compared that with a 1x1 conv projection layer (ResyBlockV2).
# After 5 epochs, the net with 1x1 con projection layer learned way faster
# achieving ~23% loss compared to 42% loss using 3x3 conv for projection.
# My pseudo-reasoning here would be that while having more parameters
# increasing the space of the approximated function, it makes it harder
# to converge to at an earlier stage. Constraining the function would definitely make it more learanable.
# Conclusion: 1x1 conv was better than 3x3 conv for projection.

# Fusion:
# (ResyBlockV3): The second thing I attempted was fusing by performing elmentwise
# maxpool between the identity branch and the branch processed by
# two conv layers. This was motivated by the thought that if the net is really
# picking specific paths for to use more for inference. Why do we still
# add a path that has been deemed not very useful?
# Surprisingly, it performed strictly better than ResyBlockV1 (3x3 conv projection and addition fused)
# but slightly worse than ResyBlockV2.
# (ResyBlockV4) : I also experimented doing some form
# of learnt fusion so I took the output from the identiy branch (x_p) and the output
# from the 2 conv branch (x), added them then passed them through another conv
# layer with 3x3 kernel size and same padding. The result were as bad as ResyBlockV1. Again
# I think the amount of training are not very fair. I'd think that it'd generalize better after
# it's been allowed to converge but from few epochs it performed badly.

# ResyBlockV1 : 72:06% Accuracy, 42% training loss @ epoch 5
# ResyBlockV2 : 77.16% Accuracy, 26% training loss @ epoch 5
# ResyBlockV3 : 77.55% Accuracy, 30% training loss @ epoch 5
# ResyBlockV4 : 74.16% Accuracy, 42% training loss @ epoch 5

import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import os
import sys
import torchvision

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


# This is V1 of a residual block. Input is processed
# through 2 conv layers with "same" padding and finally
# fused with a projection of the original input using addition.
# This version computes the projection using 3x3 conv and same padding
# Check script level docstring for rationale.
class ResyBlockV1(nn.Module):
    def __init__(self, in_chan, out_ch):
        super(ResyBlockV1, self).__init__()
        self.depth = out_ch
        self.conv3x3_same_0 = nn.Conv2d(out_channels=self.depth,
                                        in_channels=in_chan,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.conv3x3_same_1 = nn.Conv2d(out_channels=self.depth,
                                        in_channels=self.depth,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.project = nn.Conv2d(out_channels=out_ch,
                                 in_channels=in_chan,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)

        self.bn = nn.BatchNorm2d(self.depth)

    def forward(self, x):
        x_p = x
        x = F.relu(self.bn(self.conv3x3_same_0(x)))
        x = F.relu(self.bn(self.conv3x3_same_1(x)))
        out = x + self.project(x_p)
        return out


# This is V2 of a residual block. Input is processed
# through 2 conv layers with "same" padding and finally
# fused with a projection of the original input using addition.
# This version computes the projection using 1x1 conv and 0 padding
# Check script level docstring for rationale.
class ResyBlockV2(nn.Module):
    def __init__(self, in_chan, out_ch):
        super(ResyBlockV2, self).__init__()
        self.depth = out_ch
        self.conv3x3_same_0 = nn.Conv2d(out_channels=self.depth,
                                        in_channels=in_chan,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.conv3x3_same_1 = nn.Conv2d(out_channels=self.depth,
                                        in_channels=self.depth,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.project = nn.Conv2d(out_channels=out_ch,
                                 in_channels=in_chan,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.bn = nn.BatchNorm2d(self.depth)

    def forward(self, x):
        x_p = x
        x = F.relu(self.bn(self.conv3x3_same_0(x)))
        x = F.relu(self.bn(self.conv3x3_same_1(x)))
        out = x + self.project(x_p)
        return out


# This is V3 of a residual block. Input is processed
# through 2 conv layers with "same" padding and finally
# fused with a projection of the original input using addition.
# This version computes the projection using 1x1 conv and 0 padding
# Fusion is done by elementwise max pool.
# Check script level docstring for rationale.
class ResyBlockV3(nn.Module):
    def __init__(self, in_chan, out_ch):
        super(ResyBlockV3, self).__init__()
        self.depth = out_ch
        self.conv3x3_same_0 = nn.Conv2d(out_channels=self.depth,
                                        in_channels=in_chan,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.conv3x3_same_1 = nn.Conv2d(out_channels=self.depth,
                                        in_channels=self.depth,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.project = nn.Conv2d(out_channels=out_ch,
                                 in_channels=in_chan,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.bn = nn.BatchNorm2d(self.depth)

    def forward(self, x):
        x_p = x
        x = F.relu(self.bn(self.conv3x3_same_0(x)))
        x = F.relu(self.bn(self.conv3x3_same_1(x)))
        out = torch.max(x, self.project(x_p))
        return out


# This is V3 of a residual block. Input is processed
# through 2 conv layers with "same" padding and finally
# fused with a projection of the original input using addition.
# This version computes the projection using 1x1 conv and 0 padding
# Fusion is done by addition and conv layer
# Check script level docstring for rationale.
class ResyBlockV4(nn.Module):
    def __init__(self, in_chan, out_ch):
        super(ResyBlockV4, self).__init__()
        self.depth = out_ch
        self.conv3x3_same_0 = nn.Conv2d(out_channels=self.depth,
                                        in_channels=in_chan,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.conv3x3_same_1 = nn.Conv2d(out_channels=self.depth,
                                        in_channels=self.depth,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)
        self.project = nn.Conv2d(out_channels=out_ch,
                                 in_channels=in_chan,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.fuse = nn.Conv2d(out_channels=self.depth,
                              in_channels=self.depth,
                              kernel_size=3,
                              stride=1,
                              padding=1)

        self.bn = nn.BatchNorm2d(self.depth)

    def forward(self, x):
        x_p = x
        x = F.relu(self.bn(self.conv3x3_same_0(x)))
        x = F.relu(self.bn(self.conv3x3_same_1(x)))
        out = x + self.project(x_p)
        out = self.fuse(out)
        return out


class ResyNet(nn.Module):
    def __init__(self, resy_block, num_classes=10, input_res=(32, 32)):
        super(ResyNet, self).__init__()
        self.res1 = resy_block(in_chan=3, out_ch=64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2)
        # [16x16x64]
        self.res2 = resy_block(in_chan=64, out_ch=128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2)
        # [8x8x128]
        self.res3 = resy_block(in_chan=128, out_ch=256)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2)
        # [4x4x256]
        self.res4 = resy_block(in_chan=256, out_ch=512)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2)
        # [2x2x512]
        self.pre_fc1 = (input_res[0] // 2**4) * (input_res[1] // 2**4)
        self.fc1 = nn.Linear(self.pre_fc1 * 512, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.res1(x)
        x = self.max_pool1(x)
        x = self.res2(x)
        x = self.max_pool2(x)
        x = self.res3(x)
        x = self.max_pool3(x)
        x = self.res4(x)
        x = self.max_pool4(x)
        x = x.view(-1, self.pre_fc1 * 512)
        x = self.fc1(x)
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
        print("Epoch {}: {}".format(epoch,
                                    running_loss / len(loaders['train'])))
    return


def test(net, loaders):
    num_samples = 0
    num_corrects = 0
    with torch.no_grad():
        for i, data in enumerate(loaders['validation']):
            x, y = data
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            y_pred = net(x).argmax(dim=1)
            num_corrects += (y_pred == y).sum().item()
            num_samples += x.shape[0]
    print("Classification Accuracy: {:.2f}".format(num_corrects / num_samples *
                                                   100))


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

    net = ResyNet(resy_block=ResyBlockV3)
    train(net, loaders)
    test(net, loaders)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
