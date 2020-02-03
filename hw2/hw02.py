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
TRAIN_BATCH = 10
VALIDATION_BATCH = 10
CLASSES_TO_SAMPLE = ['cat', 'dog']
# NOTE: PyTorch is channels last. Given CIFAR10 order
CHAN_FIRST_ORDER = {
    'N': 0,
    'H': 2,
    'W': 3,
    'C': 1,
}


def _batch(mode):
    return TRAIN_BATCH if mode == 'train' else VALIDATION_BATCH


def _modes():
    return ['train', 'validation']


def _dim(dim_name):
    return CHAN_FIRST_ORDER[dim_name]


def preprocess_cifar10():
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])
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
            import ipdb; ipdb.set_trace()
            writer.add_image(tag='gt/{}'.format(mode),
                             img_tensor=grid,
                             global_step=step_tracker.step(mode),
                             dataformats='CHW')
            step_tracker.update(mode)
            if step_tracker.total_steps() == 100:
                print("Done. visualize via:\n tensorboard --log_dir={}".format(LOGS_DIR))
                return 


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

    _visualize_in_tensorboard(loaders)


if __name__ == "__main__":
    sys.exit(main())
