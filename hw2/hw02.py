try:
    import torchvision
    import torch
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError as e:
    print("Failed to import torchvision and torch")
    raise

import sys
import os
import time

DATASET_ROOT_DIR = os.path.expanduser("~/.data")
LOGS_DIR = os.path.expanduser('/tmp/.logs/{}'.format(
    time.strftime('%m%d%y:%H%M')))
TRAIN_BATCH = 5
VALIDATION_BATCH = 5
CLASSES_TO_SAMPLE = ['cat', 'dog']
# NOTE: PyTorch is channels last. Given CIFAR10 order
CHAN_FIRST_ORDER = {
    'N': 0,
    'H': 2,
    'W': 3,
    'C': 1,
}

NORMALIZATION_MEAN = [0.5, 0.5, 0.5]
NORMALIZATION_STD = [0.5, 0.5, 0.5]


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


class StepTracker():
    def __init__(self):
        self._train_step = 0
        self._valid_step = 0

    def update(self, mode):
        if mode == 'train':
            self._train_step += 1
        else:
            self._valid_step += 1

    def step(self, mode):
        return self._train_step if mode == 'train' else self._valid_step

    def total_steps(self):
        return self._train_step + self._valid_step


def _make_dir_with_permissions(path):
    if not os.path.exists(path):
        os.makedirs(path, mode=0o777, exist_ok=True)


def _make_writers(modes):
    writers = {}
    for mode in _modes():
        path = '{}/{}'.format(LOGS_DIR, mode)
        _make_dir_with_permissions(path)
        writers[mode] = SummaryWriter(log_dir=path, flush_secs=30)
    return writers


def _visualize_in_tensorboard(loaders):
    writers = _make_writers(list(loaders.keys()))
    step_tracker = StepTracker()
    for mode in loaders.keys():
        for imgs, target in loaders[mode]:
            writer = writers[mode]
            grid = torchvision.utils.make_grid(tensor=imgs, nrow=2)
            import ipdb
            ipdb.set_trace()
            writer.add_image(tag='gt/{}'.format(mode),
                             img_tensor=grid,
                             global_step=step_tracker.step(mode),
                             dataformats='CHW')
            step_tracker.update(mode)
            if step_tracker.total_steps() == 100:
                print("Done. visualize via:\n tensorboard --log_dir={}".format(
                    LOGS_DIR))
                return


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


def train_two_layer_network(loaders, new_classes):
    D_in, H1, H2, D_out = 3 * 32 * 32, 1000, 256, 2
    dtype = torch.float
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    w1 = torch.randn(D_in, H1, device=device, dtype=dtype)
    w2 = torch.randn(H1, H2, device=device, dtype=dtype)
    w3 = torch.randn(H2, D_out, device=device, dtype=dtype)
    LEARNING_RATE = 1e-5
    labels_postprocesser = one_hot_encode_and_filter_cifar_labels(
        loaders['train'].dataset, new_classes)
    for t in range(2500):
        for i, data in enumerate(loaders['train']):
            inputs, labels = data
            x = inputs.to(device)
            y = labels_postprocesser(labels).to(device)
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
            a3 = z3.clamp(min=0)

            import ipdb
            ipdb.set_trace()
            # [N, 2]
            loss = .5 * (a3 - y).pow(2).mean().item()
            if (i % 100) == 99:
                import ipdb
                ipdb.set_trace()
                print("L2 loss at iter {}: {}".format(i, loss))
            da3 = a3 - y
            dz3 = da3 * (z3 > 0).float()
            dw3 = a2.t().mm(dz3)

            dz2 = dz3.mm(w3.t()) * (z2 > 0).float()
            dw2 = a1.t().mm(dz2)

            dz1 = dz2.mm(w2.t()) * (z1 > 0).float()
            dw1 = x.t().mm(dz1)

            w1 -= LEARNING_RATE * dw1
            w2 -= LEARNING_RATE * dw2
            w3 -= LEARNING_RATE * dw3


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

    # _visualize_in_tensorboard(loaders)
    train_two_layer_network(loaders, CLASSES_TO_SAMPLE)


if __name__ == "__main__":
    sys.exit(main())
