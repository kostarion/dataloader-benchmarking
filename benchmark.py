import os
import numpy as np
import time
import torch

from datasets import METHODS
from genericpath import getsize
from pathlib import Path

BATCH_SIZE = 1
WORKERS = 1
PIN_MEMORY = False
PREFETCH_FACTOR = 1
REPS = 2
SEED = 42

# CUDA_LAUNCH_BLOCKING=1

def get_size(folder: str) -> int:
    return sum(p.stat().st_size for p in Path(folder).rglob('*'))

class Benchmark(object):
    def __init__(self, dataloader, reps=REPS):
        self.dataloader = dataloader
        self.data_dir = self.dataloader.dataset.root
        self.method = self.dataloader.dataset.extension
        self.reps = int(reps)
	
    def run(self):
        times = []
        print(f"\n*** Running {self.method} benchmark... ***")
        print(self.data_dir)
        print(f"Number of elements in dir: {len(self.dataloader.dataset)}")
        print(f"Dir size: {get_size(self.data_dir)/1024**2:.2f} MB")
        for k in range(self.reps):
            torch.manual_seed(SEED+k)
            start = time.time()
            for batch, _ in self.dataloader:
                batch.mean()
                times.append(time.time() - start)
        print(f"Mean time: {np.mean(times):.3f} s")

if __name__ == "__main__":
    os.environ["SAFETENSORS_FAST_GPU"] = "1"
    for extension, dataset_cls in reversed(METHODS):
        dataset = dataset_cls(extension)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=True, 
            pin_memory=PIN_MEMORY,
            # num_workers=WORKERS,
            # prefetch_factor=PREFETCH_FACTOR,
            # multiprocessing_context='spawn'
            )
        benchmark = Benchmark(dataloader)
        benchmark.run()
