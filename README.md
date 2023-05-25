## dataloader-benchmarking
# Benchmarking file formats for latents/embeddings storage

One of the possible rooms for GPU utilisation improvements when training large LDMs is to precompute image latents and CLIP embeddings and use them as a training data for UNet. Mosaic [announced](https://www.mosaicml.com/blog/stable-diffusion-part-3) that it allowed them to gain ~1.4x training speedup. Moreover, it allows to save GPU VRAM during training not having to run VAE and CLIP over each sample. 

A choice of a file format to store these embeddings might turn out to be a crucial factor for balancing IO and GPU utilisation for large datasets due to different available decoding and memory→GPU transferring mechanisms.

4 formats are being studied in this experiment: **.npy**, **.pt**, [**.safetensors**](https://huggingface.co/docs/safetensors/index), and **.bin**. In .bin format we use just a binary dump of a flattened array data, considering the shape metadata for a later restoration of the original tensor dimensions to be known.

## Results

We benchmark the speed of batched reading from files to a single A100 GPU. Single DataLoader worker is used (no multiprocessing), so no prefetch is used; pin_memory=False. Each benchmarking run is repeated 10 times and the average time is calculated.

Single .mean() operation is performed after the tensor batch is loaded to ensure that data transfer to GPU is finished.

### Experiment 1:

**100K samples** of **(1x4x32x32)** latents, each saved as a separate file. 

Reading all of them in **batches of 192** elements (~1GB of GPU VRAM allocated). 

|  | .bin | .npy | .pt | .safetensors |
| --- | --- | --- | --- | --- |
| Disk usage (MB) | 781.25 | 793.46 | 852.49 | 788.88 |
| HDD batched read (s) | 22.66 | 49.62 | 48.99 | 30.95 |
| NVME batched read (s) | 2.98 | 12.59 | 10.8 | 6.76 |
| SHM batched read (s) | 2.98 | 12.7 | 10.69 | 6.7 |

### Experiment 2:

**1000 samples** of **(10Kx4x32x32)** latents, each saved as a separate file. 

Reading all of them in **batches of 16** elements (~5GB of GPU VRAM allocated).

|  | .bin | .npy | .pt | .safetensors |
| --- | --- | --- | --- | --- |
| Disk usage (GB) | 78.12 | 78.12 | 78.12 | 78.12 |
| HDD batched read (s) | 59.6 | 60.8 | 66.69 | 55.3 |
| NVME batched read (s) | 25.28 | 25.1 | 33.94 | 12.03 |
| SHM batched read (s) | 26.67 | 25.9 | 33.95 | 12.21 |

### Experiment 3:

**10 samples** of **(1Mx4x32x32)** latents, each saved as a separate file. 

Reading all of them in **batches of 1** element (~25GB of GPU VRAM allocated).

|  | .bin | .npy | .pt | .safetensors |
| --- | --- | --- | --- | --- |
| Disk usage (GB) | 78.12 | 78.12 | 78.12 | 78.12 |
| HDD batched read (s) | 66.38 | 103.38 | 96.52 | 39.41 |
| NVME batched read (s) | 27.65 | 28.36 | 36.09 | 13.24 |
| SHM batched read (s) | 32.37 | 29.36 | 38.9 | 14.09 |

## Main takeaways:
1. When working with large files, **Safetensors** do allow to save significant amount of loading time comparing to the other methods (**> x2** comparing to .npy and .pt). The speed-up mainly comes from the effect of tensors zero-copy that allows to skip loading data to CPU.
”The library works by memory mapping the file, creating the tensor empty with pytorch and calling `cudaMemcpy` directly to move the tensor directly on the GPU.”
2. Reading from NVME and Shared memory (SHM) gives almost identical time performance. 
”This queuing mechanism lets NVMe make better use of the parallel processing capabilities of an SSD, something the other protocols cannot do. In addition, NVMe uses remote direct memory access over the PCIe bus to map I/O commands and responses directly to the host's shared memory. This reduces CPU overhead even further and improves NVMe speeds. As a result, each CPU instruction cycle can support higher IOPS and reduce latencies in the host software stack.”
3. When using small separate files, a simple binary dump outperforms all the other methods probably due to an effect of comparatively small header and insignificant time of reshape operation in this case.
