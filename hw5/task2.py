# In this task, I looked into preprocessing the input images to reduce the effect of noise.
# I tried using a gaussian filter and a mean filter. Both achieved approximately same results.
# What seemed to be more crucial for getting good results and maintaining numerical stability was normalizing the inputs.
# Diviing the input images by 255 so that the data is between 0, 1 stabilized training and with that I dont believe any smoothing is necessary but I showed results with a mean filter.

import random
import numpy as np
import torch
import torchvision
from torchvision import transforms
from skimage.filters import gaussian
from scipy.ndimage import median_filter
import cv2
from PIL import ImageFilter
import torch.nn.functional as F
import os
import sys
from torch.utils.tensorboard import SummaryWriter

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)

from DLStudio import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GaussianSmooth(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, tensor):
        img_np = np.transpose(tensor['image'].numpy(), axes=[1, 2, 0])
        tensor['image'] = torch.from_numpy(
            cv2.blur(img_np, ksize=(3, 3)).astype('float32')).float()

        tensor['image'] = tensor['image'].permute(2, 0, 1)
        tensor['image'] = tensor['image'] / 255.0
        return tensor


def print_conf_matrix(class_names, matrix, file_handle=sys.stdout):
    row_format = "{:>10}" + "{:>10.3f}" * (len(class_names))
    head_format = "{:>10}" * (len(class_names) + 1)
    print(head_format.format("", *class_names), file=file_handle)
    for class_name, row in zip(class_names, matrix):
        print(row_format.format(class_name, *row), file=file_handle)


def evaluate(net, loader):
    out_file = open("/tmp/output.txt", "a+")
    print("Task 2:", file=out_file)
    # Evaluate TripleNet
    conf_matrix = torch.zeros(size=(5, 5))
    samples_per_class = torch.zeros(size=(5, ))
    for i, data in enumerate(loader):
        x, y = data['image'], data['label']
        y_pred = torch.argmax(net(x)[0], dim=-1, keepdims=False)
        y_pred_cpu = y_pred.to(torch.device("cpu"))
        conf_matrix[y, y_pred_cpu] += 1
        samples_per_class[y] += 1

    accuracy = conf_matrix.diagonal().sum().item() / conf_matrix.sum().item()
    conf_matrix /= samples_per_class
    conf_matrix *= 100
    print("Dataset50 Classification Accuracy: {:.3f}".format(accuracy * 100),
          file=out_file)
    print("Dataset50 Confusion Matrix:", file=out_file)
    print_conf_matrix(class_names=sorted(loader.dataset.label_map,
                                         key=lambda item: item[1]),
                      matrix=conf_matrix.numpy().tolist(),
                      file_handle=out_file)
    print("", file=out_file)
    out_file.close()


def main(argv):
    dls = DLStudio(
        dataroot="/home/nhendy/DLStudio-1.1.0/Examples/data/",
        image_size=[32, 32],
        path_saved_model="./saved_model",
        momentum=0.9,
        learning_rate=1e-4,
        epochs=2,
        batch_size=5,
        classes=('rectangle', 'triangle', 'disk', 'oval', 'star'),
        debug_train=1,
        debug_test=1,
        use_gpu=True,
    )

    detector = DLStudio.DetectAndLocalize(dl_studio=dls)

    dataserver_train = DLStudio.DetectAndLocalize.PurdueShapes5Dataset(
        train_or_test='train',
        dl_studio=dls,
        dataset_file="PurdueShapes5-10000-train-noise-50.gz",
        transform=GaussianSmooth(sigma=0.5))

    dataserver_test = DLStudio.DetectAndLocalize.PurdueShapes5Dataset(
        train_or_test='test',
        dl_studio=dls,
        dataset_file="PurdueShapes5-1000-test-noise-50.gz",
        transform=GaussianSmooth(sigma=.5))

    detector.dataserver_train = dataserver_train
    detector.dataserver_test = dataserver_test

    detector.load_PurdueShapes5_dataset(dataserver_train, dataserver_test)
    detector.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           num_workers=4)

    model = detector.LOADnet2(skip_connections=True, depth=32)

    dls.show_network_summary(model)

    detector.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model)

    evaluate(model, detector.test_dataloader)


if __name__ == "__main__":
    main(sys.argv[1:])
