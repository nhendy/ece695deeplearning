# In this task I decided to implement a multitask neural network
# I structured the neural such that it has a backbone of a sequence of resblocks. The output of that backbone is used by all 3 output paths.
# One path for classfiying shapes, one for classifying noise level and
# one for bbox regression. I trained the net on the mixed dataset (40k) samples
# The results are as shown in the output.txt.
import torch
import sys
import os
import gzip
import pickle
import random
import argparse
from torch import nn
import torchvision
import numpy as np
from torch.nn import functional as F
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)

from DLStudio import DLStudio

LOADnet2 = DLStudio.DetectAndLocalize.LOADnet2

EPOCHS = 3
VALIDATION_BATCH = 5
TRAIN_BATCH = 5

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",
                        default="/home/nhendy/DLStudio-1.1.0/Examples/data",
                        type=str,
                        help="Purdue dataset root directory.")
    return parser.parse_args(argv)


def _modes():
    return ['train', 'test']


def _batch(mode):
    return TRAIN_BATCH if mode == 'train' else VALIDATION_BATCH


class PurdueMixedDataset(torch.utils.data.Dataset):
    noise_to_class = {"0": 0, "20": 1, "50": 2, "80": 3}
    class_to_noise = dict(map(reversed, noise_to_class.items()))

    def __init__(self,
                 root,
                 noise_levels=["0", "20", "50", "80"],
                 train=True,
                 transform=None,
                 shuffle=False):
        super(PurdueMixedDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train
        self.dataset = {}
        self.samples = []
        self.label_map = {}
        self.class_labels = {}
        for noise_level in noise_levels:
            if noise_level == "0":
                noise_level_tarball = "PurdueShapes5-{}-{}.gz".format(
                    "10000" if self.train else "1000",
                    "train" if self.train else "test")
            else:
                noise_level_tarball = "PurdueShapes5-{}-{}-noise-{}.gz".format(
                    "10000" if self.train else "1000",
                    "train" if self.train else "test", noise_level)
            self._maybe_cache_data(noise_level_tarball, noise_level)
        for noise in self.dataset.keys():
            for k in self.dataset[noise].keys():
                self.dataset[noise][k].append(self.noise_to_class[noise])
                self.samples.append(self.dataset[noise][k])
        if shuffle:
            random.shuffle(self.samples)

    def _maybe_cache_data(self, tarball, noise_key):
        basename, _ = os.path.splitext(tarball)
        if os.path.exists("torch-saved-{}-dataset.pt".format(basename)) and \
                  os.path.exists("torch-saved-{}-label-map.pt".format(basename)):
            print("\nLoading training data from the torch-saved archive")
            self.dataset[noise_key] = torch.load(
                "torch-saved-{}-dataset.pt".format(basename))
            self.label_map = torch.load(
                "torch-saved-{}-label-map.pt".format(basename))
        else:
            print(
                """\n\n\nLooks like this is the first time you will be loading in\n"""
                """the dataset for this script. First time loading could take\n"""
                """a minute or so.  Any subsequent attempts will only take\n"""
                """a few seconds.\n\n\n""")
            f = gzip.open(os.path.join(self.root, tarball), 'rb')
            dataset = f.read()
            if sys.version_info[0] == 3:
                self.dataset[noise_key], self.label_map = pickle.loads(
                    dataset, encoding='latin1')
            else:
                self.dataset[noise_key], self.label_map = pickle.loads(dataset)
            torch.save(self.dataset[noise_key],
                       "torch-saved-{}-dataset.pt".format(basename))
            torch.save(self.label_map,
                       "torch-saved-{}-label-map.pt".format(basename))
        # reverse the key-value pairs in the label dictionary:
        self.class_labels = dict(map(reversed, self.label_map.items()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r = np.array(self.samples[idx][0])
        g = np.array(self.samples[idx][1])
        b = np.array(self.samples[idx][2])
        R, G, B = r.reshape(32, 32), g.reshape(32, 32), b.reshape(32, 32)
        im_tensor = torch.zeros(3, 32, 32, dtype=torch.float)
        im_tensor[0, :, :] = torch.from_numpy(R)
        im_tensor[1, :, :] = torch.from_numpy(G)
        im_tensor[2, :, :] = torch.from_numpy(B)
        bb_tensor = torch.tensor(self.samples[idx][3], dtype=torch.float)
        sample = {
            'image': im_tensor,
            'bbox': bb_tensor,
            'label': self.samples[idx][4],
            'noise': self.samples[idx][-1]
        }
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample


# Credit: This is adapted from my hw4 submission. It's used as a
# a building block for NoiseNet
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

        self.bn_1 = nn.BatchNorm2d(self.depth)
        self.bn_2 = nn.BatchNorm2d(self.depth)

    def forward(self, x):
        x_p = x
        x = F.relu(self.bn_1(self.conv3x3_same_0(x)))
        x = F.relu(self.bn_2(self.conv3x3_same_1(x)))
        out = x + self.project(x_p)
        return out


# This network is inspired by resnet architecture
class TripleNet(nn.Module):
    def __init__(self,
                 res_block=ResyBlockV2,
                 shape_num_classes=5,
                 noise_num_classes=4,
                 input_res=(32, 32)):
        super(TripleNet, self).__init__()
        self.layers = []
        self.blocks = 3
        channels_last = 3
        for i in range(self.blocks):
            self.layers.append(
                res_block(in_chan=channels_last, out_ch=64 * 2**i))
            self.layers.append(nn.MaxPool2d(kernel_size=2))
            channels_last = 64 * 2**i
        self.backbone = nn.Sequential(*self.layers)
        self.pre_fc1 = (input_res[0] // 2**self.blocks) * (input_res[1] //
                                                           2**self.blocks)
        self.fc1 = nn.Linear(self.pre_fc1 * 64 * 2**(self.blocks - 1), 1000)
        self.shape_classifier = nn.Linear(1000, shape_num_classes)
        self.noise_classifier = nn.Linear(1000, noise_num_classes)

        self.regressor = nn.Sequential(
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      padding=1), nn.ReLU(inplace=True), nn.Flatten(),
            nn.Linear(self.pre_fc1 * 64, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 512), nn.Linear(512, 4))

    def forward(self, x):
        backbone = self.backbone(x)
        # import ipdb
        # ipdb.set_trace()
        x = backbone.view(-1, self.pre_fc1 * 64 * 2**(self.blocks - 1))
        fc1 = self.fc1(x)
        return self.shape_classifier(fc1), self.noise_classifier(
            fc1), self.regressor(backbone)


def train(net, loaders, lr=1e-4, momentum=0.9):
    net = net.to(DEVICE)
    xentropy_loss = torch.nn.CrossEntropyLoss()
    msee_loss = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    for epoch in range(EPOCHS):
        running_loss = {
            k: {
                'train': 0.0,
                'test': 0.0
            }
            for k in ['xentropy', 'msee']
        }
        for mode in _modes():
            if mode == "test":
                net.eval()
            else:
                net.train()
            for i, data in enumerate(loaders[mode]):
                inputs, shape_labels, gt_bbox, noise_labels = data[
                    'image'], data['label'], data['bbox'], data['noise']
                inputs = inputs.to(DEVICE)
                shape_labels = shape_labels.to(DEVICE)
                noise_labels = noise_labels.to(DEVICE)
                gt_bbox = gt_bbox.to(DEVICE)
                optimizer.zero_grad()
                shape_class_logits, noise_class_logits, pred_bbox = net(inputs)
                shape_classification_loss = xentropy_loss(
                    shape_class_logits, shape_labels)
                noise_classification_loss = xentropy_loss(
                    noise_class_logits, noise_labels)
                regression_loss = msee_loss(pred_bbox, gt_bbox)
                classification_loss = shape_classification_loss + noise_classification_loss
                total_loss = classification_loss + regression_loss
                running_loss['xentropy'][mode] += classification_loss.item()
                running_loss['msee'][mode] += regression_loss.item()
                if mode == "train":
                    total_loss.backward()
                    optimizer.step()
                if mode == "train" and i % 500 == 0:
                    print(
                        "Training Noise {}, loss, iter {:04d}, Running: Xentropy {:.3f} L2 loss {:.3f}, Current: [{:.3f},{:.3f}]"
                        .format(data['noise'][0].item(), i,
                                running_loss['xentropy']['train'] / (i + 1),
                                running_loss['msee']['train'] / (i + 1),
                                classification_loss.item(),
                                regression_loss.item()))

        print(
            "Epoch {:03d}: Train Loss [{:.3f}, {:.3f}], Validation Loss [{:.3f}, {:.3f}]"
            .format(epoch,
                    running_loss['xentropy']['train'] / len(loaders['train']),
                    running_loss['msee']['train'] / len(loaders['train']),
                    running_loss['xentropy']['test'] / len(loaders['test']),
                    running_loss['msee']['test'] / len(loaders['test'])))


def train_triplenet(data_root):
    transforms = torchvision.transforms.Compose([lambda x: x / 255.0])
    loaders = {}
    for mode in _modes():
        dataset = PurdueMixedDataset(root=data_root,
                                     train=mode == 'train',
                                     transform=transforms,
                                     shuffle=True)
        loaders[mode] = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=_batch(mode),
                                                    drop_last=True)
    net = TripleNet()
    print("Training TripleNet")
    print(net)
    train(net, loaders)
    return net


def print_conf_matrix(class_names, matrix, file_handle=sys.stdout):
    row_format = "{:>10}" + "{:>10.3f}" * (len(class_names))
    head_format = "{:>10}" * (len(class_names) + 1)
    print(head_format.format("", *class_names), file=file_handle)
    for class_name, row in zip(class_names, matrix):
        print(row_format.format(class_name, *row), file=file_handle)


def evaluate(data_root, net):
    out_file = open("output.txt", "a+")
    print("Task 4:")
    transforms = torchvision.transforms.Compose([lambda x: x / 255.0])
    dataset = PurdueMixedDataset(root=data_root,
                                 train=False,
                                 transform=transforms,
                                 noise_levels=['50'],
                                 shuffle=True)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=1,
                                         drop_last=True)
    # Evaluate TripleNet
    triplenet_conf_matrix = torch.zeros(size=(5, 5))
    samples_per_class = torch.zeros(size=(5, ))
    for i, data in enumerate(loader):
        x, y = data['image'], data['label']
        x = x.to(DEVICE)
        y_pred = torch.argmax(net(x)[0], dim=-1, keepdims=False)
        y_pred_cpu = y_pred.to(torch.device("cpu"))
        triplenet_conf_matrix[y, y_pred_cpu] += 1
        samples_per_class[y] += 1

    triplenet_accuracy = triplenet_conf_matrix.diagonal().sum().item(
    ) / triplenet_conf_matrix.sum().item()
    triplenet_conf_matrix /= samples_per_class
    triplenet_conf_matrix *= 100
    print("Dataset50 Classification Accuracy: {:.3f}".format(
        triplenet_accuracy * 100),
          file=out_file)
    print("Dataset50 Confusion Matrix:", file=out_file)
    print_conf_matrix(class_names=sorted(dataset.label_map,
                                         key=lambda item: item[1]),
                      matrix=triplenet_conf_matrix.numpy().tolist(),
                      file_handle=out_file)
    print("", file=out_file)
    out_file.close()


def main(argv):
    args = parse_args(argv)
    triplenet = train_triplenet(args.root)
    evaluate(args.root, triplenet)


if __name__ == "__main__":
    main(sys.argv[1:])
