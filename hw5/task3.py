# In this task I trained a total of 5 networks one LOADnet per noise level and a noise classifier network.
# Noise classification is relatively a trivial task since the noise injected into the dataset
# was likely very structured so it should be easy for a neural network to figure it out.
# That indeed was the case, training converged pretty quickly and a near perfect accuracy was achieved on validation set.

# Based on the noise class, the image is redirected to a classfier that was
# trained on the predicted noise level.
# A more end to end approach is presented in task4 with TripleNet
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

EPOCHS = 5
VALIDATION_BATCH = 5
TRAIN_BATCH = 4

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
class NoiseNet(nn.Module):
    def __init__(self,
                 res_block=ResyBlockV2,
                 num_classes=10,
                 input_res=(32, 32)):
        super(NoiseNet, self).__init__()
        self.layers = []
        self.blocks = 4
        channels_last = 3
        for i in range(self.blocks):
            self.layers.append(
                res_block(in_chan=channels_last, out_ch=64 * 2**i))
            self.layers.append(nn.MaxPool2d(kernel_size=2))
            channels_last = 64 * 2**i
        self.backbone = nn.Sequential(*self.layers)
        self.pre_fc1 = (input_res[0] // 2**self.blocks) * (input_res[1] //
                                                           2**self.blocks)
        self.fc1 = nn.Linear(self.pre_fc1 * 512, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, self.pre_fc1 * 512)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train_classifier(net, loaders, epochs, input_label_fn):
    net = net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for epoch in range(epochs):
        running_loss = {'train': 0.0, 'test': 0.0}
        running_tps = {'train': 0, 'test': 0}
        running_samples = {'train': 0, 'test': 0}
        for mode in _modes():
            if mode == "test":
                net.eval()
            else:
                net.train()
            for i, data in enumerate(loaders[mode]):
                inputs, labels = input_label_fn(data)
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_tps[mode] += (outputs.argmax(
                    dim=1) == labels).sum().item()
                running_samples[mode] += inputs.shape[0]
                running_loss[mode] += loss.item()
                if mode == "train":
                    loss.backward()
                    optimizer.step()
                if mode == "train" and i % 500 == 0:
                    print("Training loss, iter {}: {}".format(
                        i, running_loss['train'] / (i + 1)))

        print(
            "Epoch {}: Train Loss {}, Validation Loss {}. Train Accuracy {}, Validation Accuracy {}"
            .format(epoch, running_loss['train'] / len(loaders['train']),
                    running_loss['test'] / len(loaders['test']),
                    running_tps['train'] / running_samples['train'],
                    running_tps['test'] / running_samples['test']))


def train_noise_net(data_root):
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
    net = NoiseNet(num_classes=4)
    print("Training NoiseNet")
    print(net)
    train_classifier(net, loaders, 1, lambda data:
                     (data['image'], data['noise']))
    return net


def train_detector(net, loaders, lr=1e-4, momentum=0.9):
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
        running_tps = {'train': 0, 'test': 0}
        running_samples = {'train': 0, 'test': 0}
        for mode in ['train']:
            if mode == "test":
                net.eval()
            else:
                net.train()
            for i, data in enumerate(loaders[mode]):
                inputs, labels, gt_bbox = data['image'], data['label'], data[
                    'bbox']
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                gt_bbox = gt_bbox.to(DEVICE)
                optimizer.zero_grad()
                class_logits, pred_bbox = net(inputs)
                classification_loss = xentropy_loss(class_logits, labels)
                regression_loss = msee_loss(pred_bbox, gt_bbox)
                total_loss = classification_loss + regression_loss
                running_tps[mode] += (class_logits.argmax(
                    dim=1) == labels).sum().item()
                running_samples[mode] += inputs.shape[0]
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

        print("Epoch {:03d}: Train Loss [{:.3f}, {:.3f}]".format(
            epoch,
            running_loss['xentropy']['train'] / len(loaders['train']),
            running_loss['msee']['train'] / len(loaders['train']),
            # running_loss['xentropy']['test'] / len(loaders['test']),
            # running_loss['msee']['test'] / len(loaders['test']),
            running_tps['train'] / running_samples['train']))
        # running_tps['test'] / running_samples['test']))


def train_detectors(data_root):
    noise_to_detector = {
        '80': LOADnet2(skip_connections=True, depth=10),
        '50': LOADnet2(skip_connections=True, depth=20),
        '20': LOADnet2(skip_connections=False, depth=10),
        '0': LOADnet2(skip_connections=False, depth=10)
    }
    transforms = torchvision.transforms.Compose([lambda x: x / 255.0])
    for noise_level, net in noise_to_detector.items():
        loaders = {}
        for mode in _modes():
            dataset = PurdueMixedDataset(root=data_root,
                                         train=mode == 'train',
                                         noise_levels=[noise_level],
                                         transform=transforms,
                                         shuffle=True)
            loaders[mode] = torch.utils.data.DataLoader(
                dataset=dataset, batch_size=_batch(mode), drop_last=True)
        print("Training on noise level {}".format(noise_level))
        print("-" * 50)
        print(net)
        train_detector(net, loaders)
    return noise_to_detector


def print_conf_matrix(class_names, matrix, file_handle=sys.stdout):
    row_format = "{:>10}" + "{:>10.3f}" * (len(class_names))
    head_format = "{:>10}" * (len(class_names) + 1)
    print(head_format.format("", *class_names), file=file_handle)
    for class_name, row in zip(class_names, matrix):
        print(row_format.format(class_name, *row), file=file_handle)


def run_noisenet_detector_pipeline(noisenet, noise_to_detector, x):
    noise_levels = torch.argmax(noisenet(x), dim=-1, keepdims=False)
    y = torch.zeros(size=(noise_levels.shape[0], ))
    for i, noise_level in enumerate(noise_levels):
        detector = noise_to_detector[PurdueMixedDataset.class_to_noise[
            noise_level.item()]]
        classes, _ = detector(x[i].unsqueeze(0))
        y[i] = classes.argmax(dim=-1)

    return y


def evaluate_noisenet_detector_pipeline(data_root, noise_to_detector,
                                        noise_classifier):
    out_file = open("output.txt", "a+")
    print("Task 3:", file=out_file)
    transforms = torchvision.transforms.Compose([lambda x: x / 255.0])
    dataset = PurdueMixedDataset(root=data_root,
                                 train=False,
                                 transform=transforms,
                                 shuffle=True)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=1,
                                         drop_last=True)
    # Evaluate NoiseNet
    noisenet_conf_matrix = torch.zeros(size=(4, 4))
    samples_per_class = torch.zeros(size=(4, ))
    for i, data in enumerate(loader):
        x, y = data['image'], data['noise']
        x = x.to(DEVICE)
        y_pred = torch.argmax(noise_classifier(x), dim=-1, keepdims=False)
        y_pred_cpu = y_pred.to(torch.device("cpu"))
        noisenet_conf_matrix[y, y_pred_cpu] += 1
        samples_per_class[y] += 1

    noisenet_accuracy = noisenet_conf_matrix.diagonal().sum().item(
    ) / noisenet_conf_matrix.sum().item()
    noisenet_conf_matrix /= samples_per_class
    noisenet_conf_matrix *= 100
    print("Noise Classification Accuracy: {:.3f}".format(noisenet_accuracy *
                                                         100),
          file=out_file)
    print("Noise Confusion Matrix:", file=out_file)
    print_conf_matrix(class_names=["0", "20", "50", "80"],
                      matrix=noisenet_conf_matrix.numpy().tolist(),
                      file_handle=out_file)
    print("", file=out_file)
    for noise_level in ['0', '20', '50', '80']:
        dataset = PurdueMixedDataset(root=data_root,
                                     train=False,
                                     noise_levels=[noise_level],
                                     transform=transforms,
                                     shuffle=True)
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=1,
                                             drop_last=True)
        pipeline_conf_matrix = torch.zeros(size=(5, 5))
        samples_per_class = torch.zeros(size=(5, ))
        for i, data in enumerate(loader):
            x, y = data['image'], data['label']
            x = x.to(DEVICE)
            y_pred = run_noisenet_detector_pipeline(noise_classifier,
                                                    noise_to_detector, x)
            y_pred_cpu = y_pred.to(torch.device("cpu"))
            pipeline_conf_matrix[y, y_pred_cpu.long()] += 1
            samples_per_class[y] += 1

        pipeline_accuracy = pipeline_conf_matrix.diagonal().sum().item(
        ) / pipeline_conf_matrix.sum().item()
        pipeline_conf_matrix /= samples_per_class
        pipeline_conf_matrix *= 100
        print("Dataset{} Classification Accuracy: {:.3f}".format(
            noise_level, pipeline_accuracy * 100),
              file=out_file)
        print("Dataset{} Confusion Matrix:".format(noise_level), file=out_file)
        print_conf_matrix(class_names=sorted(dataset.label_map,
                                             key=lambda item: item[1]),
                          matrix=pipeline_conf_matrix.numpy().tolist(),
                          file_handle=out_file)
        print("", file=out_file)
    out_file.close()


def main(argv):
    args = parse_args(argv)
    noise_classifier = train_noise_net(args.root)
    noise_to_detector = train_detectors(args.root)
    evaluate_noisenet_detector_pipeline(args.root, noise_to_detector,
                                        noise_classifier)


if __name__ == "__main__":
    main(sys.argv[1:])
