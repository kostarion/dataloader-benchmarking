import numpy as np
import os
import torch

from datasets import METHODS, DATA_ROOT_PATH, EMBEDDING_SHAPE, get_fname

N_SAMPLES = 10


def _get_random_tensor():
    return torch.rand(EMBEDDING_SHAPE).half()


def spawn():
    if not os.path.exists(DATA_ROOT_PATH):
        os.makedirs(DATA_ROOT_PATH)
    for extension, _ in METHODS:
        if not os.path.exists(os.path.join(DATA_ROOT_PATH, extension)):
            os.makedirs(os.path.join(DATA_ROOT_PATH, extension))

    for i in range(N_SAMPLES):
        t = _get_random_tensor()
        for extension, method in METHODS:
            method.save_tensor(t, get_fname(extension, i))


if __name__ == "__main__":
    spawn()
