import torch.utils.data as data
from PIL import Image
import numpy as np

from paddle.vision.datasets import Cifar10 as CIFAR10
from paddle.vision.datasets import Cifar100 as CIFAR100

import os
import os.path
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass
