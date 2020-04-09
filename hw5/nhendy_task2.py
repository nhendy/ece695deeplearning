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


class GaussianSmooth(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, tensor):
        # img_np = np.transpose(tensor['image'].numpy(), axes=[1, 2, 0])
        # tensor['image'] = torch.from_numpy(
        #     cv2.blur(img_np, ksize=(3, 3)).astype('float32')).float()

        # tensor['image'] = tensor['image'].permute(2, 0, 1)
        tensor['image'] = tensor['image'] / 255.0
        return tensor


def main(argv):
    dls = DLStudio(
        dataroot="/home/nhendy/DLStudio-1.1.0/Examples/data/",
        image_size=[32, 32],
        path_saved_model="./saved_model",
        momentum=0.9,
        learning_rate=1e-4,
        epochs=2,
        batch_size=4,
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

    model = detector.LOADnet2(skip_connections=True, depth=10)

    dls.show_network_summary(model)

    detector.run_code_for_training_with_CrossEntropy_and_MSE_Losses(model)

    # import pymsgbox
    # response = pymsgbox.confirm(
    #     "Finished training.  Start testing on unseen data?")
    # if response == "OK":
    detector.run_code_for_testing_detection_and_localization(model)


if __name__ == "__main__":
    main(sys.argv[1:])
