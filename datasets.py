import os
import numpy as np
import torch

from safetensors import safe_open
from safetensors.torch import load_file, save_file

DATA_ROOT_PATH = "/admin/home-dmitry/dummy_embeddings_1000000"
# DATA_ROOT_PATH = "/scratch/tmp/dummy_embeddings_1000000"
# DATA_ROOT_PATH = "/dev/shm/dummy_embeddings_10000"
EMBEDDING_SHAPE = (1000000, 4, 32, 32)
DEVICE = 'cuda:0'
NPY_ALLOW_PICKLE = False


def get_fname(extension, index):
    return os.path.join(DATA_ROOT_PATH, extension, f"{index:06d}.{extension}")


class FileFormatDataset(torch.utils.data.Dataset):
    def __init__(self, extension, device=DEVICE,
                 transform=None, target_transform=None):
        self.root = os.path.join(DATA_ROOT_PATH, extension)
        self.device = device
        self.extension = extension
        self.transform = transform
        self.target_transform = target_transform

        self.samples = []
        for fname in os.listdir(self.root):
            if fname.endswith(self.extension):
                self.samples.append(fname)

    def __getitem__(self, index):
        fname = self.samples[index]
        path = os.path.join(self.root, fname)
        sample = self._load(path)
        target = 0
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
    
    def _load(self, path):
        raise NotImplementedError
    
    def save_tensor(tensor, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


# Pytorch dataset that loads data from a directory of .npy files
class NpyDataset(FileFormatDataset):
    def _load(self, path):
        return torch.from_numpy(
            np.load(path, allow_pickle=NPY_ALLOW_PICKLE)).to(self.device)

    def save_tensor(tensor, fname):
        np.save(fname, tensor.cpu().numpy(), allow_pickle=NPY_ALLOW_PICKLE)


# Pytorch dataset that loads data from a directory of .pt files
class PtDataset(FileFormatDataset):
    def _load(self, path):
        return torch.load(path, map_location=self.device)

    def save_tensor(tensor, fname):
        torch.save(tensor, fname)


# Pytorch dataset that loads data from a directory of safetensors files
class SafetensorsDataset(FileFormatDataset):
    def _load(self, path):
        # with safe_open(path, "rb") as f:
        #     return torch.load(f)
        return load_file(path, device=self.device)["embedding"]

    def save_tensor(tensor, fname):
        save_file({"embedding": tensor}, fname)


# Pytorch dataset that loads data from a directory of binary files
class BinaryDataset(FileFormatDataset):
    def _load(self, path):
        return torch.from_numpy(
            np.fromfile(path, dtype=np.float16).reshape(EMBEDDING_SHAPE)).to(self.device)

    def save_tensor(tensor, fname):
        tensor.numpy().flatten().tofile(fname)


METHODS = (
    ("npy", NpyDataset),
    ("pt", PtDataset),
    ("safetensors", SafetensorsDataset),
    ("bin", BinaryDataset),
)
