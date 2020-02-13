try:
    import torchvision
    import torch
except ModuleNotFoundError as e:
    print("Failed to import torchvision and torch")
    raise

import sys
import os
import time
import matplotlib.pyplot as plt
import numpy as np

DATASET_ROOT_DIR = os.path.expanduser("./data")
LOGS_DIR = os.path.expanduser('/tmp/.logs/{}'.format(
    time.strftime('%m%d%y:%H%M')))
TRAIN_BATCH = 5
VALIDATION_BATCH = 5
CLASSES_TO_SAMPLE = ['cat', 'dog']
# NOTE: PyTorch is channels first. Given CIFAR10 order
CHAN_FIRST_ORDER = {
    'N': 0,
    'H': 2,
    'W': 3,
    'C': 1,
}

NORMALIZATION_MEAN = [0.5, 0.5, 0.5]
NORMALIZATION_STD = [0.5, 0.5, 0.5]
PATH_OUTPUT_PLOT = os.path.expanduser('~/.logs/plots')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(99999)

if not os.path.exists(PATH_OUTPUT_PLOT):
    os.makedirs(PATH_OUTPUT_PLOT)


def _batch(mode):
    return TRAIN_BATCH if mode == 'train' else VALIDATION_BATCH


def _modes():
    return ['train', 'validation']


def _dim(dim_name):
    return CHAN_FIRST_ORDER[dim_name]


def preprocess_cifar10():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
    ])
    return transforms


class FilterSampler(torch.utils.data.Sampler):
    def __init__(self, cifar_ds, classes_to_sample):
        self._cifar_ds = cifar_ds
        self._classes_to_sample = classes_to_sample
        self._validate_args()
        self._idx_to_sample = [
            self._cifar_ds.class_to_idx[c] for c in self._classes_to_sample
        ]

    def __iter__(self):
        for i in range(len(self._cifar_ds)):
            img, target_idx = self._cifar_ds[i]
            if target_idx not in self._idx_to_sample:
                continue
            # The sampler returns indices to iterate over not the img, target pair
            yield i

    def _validate_args(self):
        if not isinstance(self._cifar_ds, torchvision.datasets.CIFAR10):
            raise ValueError(
                "This is not a  torchvision.datasets.CIFAR10 object")
        for cls in self._classes_to_sample:
            if cls not in self._cifar_ds.class_to_idx.keys():
                raise ValueError(
                    "{} not in CIFAR10 classes. Here are the available classes:\n {}"
                    .format(cls, list(self._cifar_ds.class_to_idx)))

    @property
    def classes_to_sample(self):
        return self._classes_to_sample


def _to_one_hot(tensor, total_num_classes):
    zeros_one_hot_shaped_tensor = tensor.new_zeros(size=(tensor.size(0),
                                                         total_num_classes))
    zeros_one_hot_shaped_tensor.scatter_(
        dim=1,
        index=tensor.unsqueeze(1),
        src=tensor.new_ones(size=(tensor.size(0), 1)))
    return zeros_one_hot_shaped_tensor


def _compute_old_idx_to_new_idx(dataset, new_classes):
    classes_to_old_idx = dataset.class_to_idx
    old_idx_to_new_idx = {
        old_idx: new_classes.index(c)
        for c, old_idx in classes_to_old_idx.items() if c in new_classes
    }
    return old_idx_to_new_idx


def one_hot_encode_and_filter_cifar_labels(dataset, new_classes):
    old_idx_to_new_idx = _compute_old_idx_to_new_idx(dataset, new_classes)

    def transform(labels):
        for i, label in enumerate(labels):
            labels[i] = old_idx_to_new_idx[label.item()]
        return _to_one_hot(labels,
                           len(old_idx_to_new_idx)).to(dtype=torch.float32)

    return transform


def forward(x, state_dict):
    x = x.view(x.size(0), -1)
    # layer 1
    z1 = x.mm(state_dict['w1'])
    a1 = z1.clamp(min=0)
    # layer 2
    z2 = a1.mm(state_dict['w2'])
    a2 = z2.clamp(min=0)
    # output layer
    z3 = a2.mm(state_dict['w3'])
    return z3


def test_and_compute_accuracy(state_dict, new_classes, loaders):
    running_true_positives_sum = 0
    running_total_samples = 0
    labels_postprocesser = one_hot_encode_and_filter_cifar_labels(
        loaders['validation'].dataset, new_classes)
    for batch_idx, (x, y) in enumerate(loaders['validation']):
        x = x.to(DEVICE)
        y = labels_postprocesser(y).to(DEVICE)
        predictions = torch.argmax(forward(x, state_dict), dim=1)
        labels = torch.argmax(y, dim=1)
        running_true_positives_sum += (labels == predictions).float().sum()
        running_total_samples += x.size(0)

    accuracy = (running_true_positives_sum / running_total_samples) * 100
    print('Test Accuracy : {:.5f}'.format(accuracy))


def train_two_layer_network(loaders, new_classes, output_plot_path):
    D_in, H1, H2, D_out = 3 * 32 * 32, 1000, 256, 2
    dtype = torch.float
    w1 = torch.randn(D_in, H1, device=DEVICE, dtype=dtype)
    w2 = torch.randn(H1, H2, device=DEVICE, dtype=dtype)
    w3 = torch.randn(H2, D_out, device=DEVICE, dtype=dtype)
    LEARNING_RATE = 1e-8
    EPOCHS = 50
    labels_postprocesser = one_hot_encode_and_filter_cifar_labels(
        loaders['train'].dataset, new_classes)
    losses_per_epoch = []
    for epoch in range(EPOCHS):
        for i, data in enumerate(loaders['train']):
            inputs, labels = data
            x = inputs.to(DEVICE)
            y = labels_postprocesser(labels).to(DEVICE)
            # [N, 3072]
            x = x.view(x.size(0), -1)
            # [N, 1000]
            z1 = x.mm(w1)
            a1 = z1.clamp(min=0)
            # [N, 256]
            z2 = a1.mm(w2)
            a2 = z2.clamp(min=0)
            # [N, 2]
            z3 = a2.mm(w3)
            # [N, 2]
            loss = (z3 - y).pow(2).sum().item()

            # Backprop
            dz3 = 2 * (z3 - y)
            dw3 = a2.t().mm(dz3)

            dz2 = 2 * dz3.mm(w3.t()) * (z2 > 0).float()
            dw2 = a1.t().mm(dz2)

            dz1 = 2 * dz2.mm(w2.t()) * (z1 > 0).float()
            dw1 = x.t().mm(dz1)

            w1 -= LEARNING_RATE * dw1
            w2 -= LEARNING_RATE * dw2
            w3 -= LEARNING_RATE * dw3
        print("Epoch {}: {:.5f}".format(epoch, loss))
        losses_per_epoch.append(loss)

    print()
    plt.scatter(np.arange(len(losses_per_epoch)), losses_per_epoch)
    plt.xticks(np.arange(0, len(losses_per_epoch), 5.0))
    plt.ylabel('L2 Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.savefig(
        os.path.join(output_plot_path,
                     '{}.png'.format(time.strftime('%m%d%y:%H%M'))))

    return {'w1': w1, 'w2': w2, 'w3': w3}


def main():
    loaders = {}
    for mode in _modes():
        dataset = torchvision.datasets.CIFAR10(root=DATASET_ROOT_DIR,
                                               train=mode == 'train',
                                               transform=preprocess_cifar10(),
                                               download=True)
        sampler = FilterSampler(dataset, CLASSES_TO_SAMPLE)
        loaders[mode] = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=_batch(mode),
                                                    sampler=sampler,
                                                    drop_last=True)

    state_dict = train_two_layer_network(loaders, CLASSES_TO_SAMPLE,
                                         PATH_OUTPUT_PLOT)
    test_and_compute_accuracy(state_dict, CLASSES_TO_SAMPLE, loaders)


if __name__ == "__main__":
    sys.exit(main())
