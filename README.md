## dataloader-benchmarking
# Benchmarking file formats for latents/embeddings storage

One of the possible rooms for GPU utilisation improvements when training large LDMs is to precompute image latents and CLIP embeddings and use them as a training data for UNet. Mosaic [announced](https://www.mosaicml.com/blog/stable-diffusion-part-3) that it allowed them to gain ~1.4x training speedup. Moreover, it allows to save GPU VRAM during training not having to run VAE and CLIP over each sample. 

A choice of a file format to store these embeddings might turn out to be a crucial factor for balancing IO and GPU utilisation for large datasets due to different available decoding and memory→GPU transferring mechanisms.

4 formats are being studied in this experiment: **.npy**, **.pt**, [**.safetensors**](https://huggingface.co/docs/safetensors/index), and **.bin**. In .bin format we use just a binary dump of a flattened array data, considering the shape metadata for a later restoration of the original tensor dimensions to be known.
