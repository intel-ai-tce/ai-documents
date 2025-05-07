## Intel® Extension for OpenXLA\*

OpenXLA is an open-source Machine Learning compiler ecosystem co-developed by AI/ML industry leaders that lets developers compile and optimize models from all popular ML frameworks on a wide variety of hardware. We are pleased to announce [Intel® Extension for OpenXLA\*](https://github.com/intel/intel-extension-for-openxla), which seamlessly runs AI/ML models on Intel GPUs. Intel® Extension for OpenXLA\* is a high-performance deep learning extension implementing the [OpenXLA](https://github.com/openxla/xla) PJRT C API (see [RFC](https://github.com/openxla/community/blob/main/rfcs/20230123-pjrt-plugin.md) for details), allowing multiple AI frameworks to compile StableHLO, an operation set for high-level operations for numeric computation, as well as dispatching the executable to Intel GPUs. 

<br><div align=center><img src="/content/dam/developer/articles/technical/accelerate-stable-diffusion-on-intel-gpus-with-intel-extension-for-openxla/Intel-Data-Center-GPU-Max-Series-1100-PCIe-Card.png"></div>

<p align="center">Figure 1: Intel® Data Center GPU Max Series 1100 PCIe Card</p>

The PJRT plugin for Intel GPU is based on LLVM + SPIR-V IR code-gen technique.  It integrates with optimizations in oneAPI-powered libraries, such as Intel® oneAPI Deep Neural Network Library (oneDNN) and Intel® oneAPI Math Kernel Library (oneMKL). With optimizations on linear algebra, operator fusion, layout propagation, etc., developers can speed up their workloads without any device-specific codes. JAX is the first supported front-end. Intel® Extension for OpenXLA\* enables and accelerates training and inference of different scales workloads on Intel GPUs, including Large Language Models (LLMs) or multi-modal models, etc., such as Stable Diffusion.

<br><div align=center><img src="/content/dam/developer/articles/technical/accelerate-stable-diffusion-on-intel-gpus-with-intel-extension-for-openxla/Intel-Extension-for-OpenXLA-Architecture%20.png"></div>

<p align="center">Figure 2: Intel® Extension for OpenXLA* Architecture</p>

## Enabling Intel® Extension for OpenXLA\* 

In this section, we show how to enable JAX applications on Intel GPUs with Intel® Extension for OpenXLA\*.

### Installation

1.	Preinstallation Requirements

   * Install Ubuntu\* 22.04 OS

   * Install Intel GPU driver

      Intel® Data Center GPU Max Series [627.7 driver](https://dgpu-docs.intel.com/releases/stable_627_7_20230526.html)

   * Install Intel® oneAPI Base Toolkit

      Install oneAPI Base Toolkit Packages [2023.1](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html)

   * Setup environment variables
   ```bash
   source /opt/intel/oneapi/setvars.sh  # Modify based on your oneAPI installation directory
   ```
   * Install bazel
   ```bash
   wget https://github.com/bazelbuild/bazel/releases/download/5.3.0/bazel-5.3.0-installer-linux-x86_64.sh
   bash bazel-5.3.0-installer-linux-x86_64.sh --user
   bazel –-version   # Verify bazel version is 5.3.0
   ```
2.	Create a Python\* Virtual Environment (using Miniconda)

   * Install [Miniconda\*](https://docs.conda.io/en/latest/miniconda.html)

   * Create and activate virtual running environment
   ```bash
   conda create -n jax python=3.9
   conda activate jax
   pip install --upgrade pip
   ```
3. Build and install JAX and Intel® Extension for OpenXLA
   ```bash
   git clone https://github.com/intel/intel-extension-for-openxla.git
   cd intel-extension-for-openxla
   pip install jax==0.4.7 jaxlib==0.4.7
   ./configure
   bazel build //xla: pjrt_plugin_xpu.so
   ```
4. Set library paths
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/python-path/lib/python3.9/site-packages/jaxlib/"
   export PJRT_NAMES_AND_LIBRARY_PATHS="xpu:/xla-path/bazel-bin/xla/pjrt_plugin_xpu.so"
   ```
### Check That PJRT Intel GPU Plugin Is Loaded

Use a Python call to `jax.local_devices()` to check all available XPU devices:
```bash
python -c "import jax;print(jax.local_devices())"
```
Sample output: on Intel® Data Center GPU Max 1550 GPU (with 2 stacks):
```
[xpu(id=0), xpu(id=1)]
```
In this case of a server with single Intel® Data Center GPU Max 1550 GPU installed, there are two XPU devices, representing 2 stacks loaded into the current process. 

>**Note**: If there is no XPU devices detected, the output would be as below:
```
No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
[CpuDevice(id=0)]
```
In this case, make sure both the environment variables for `oneAPI` and the library paths for `jaxlib` as well as `PJRT plugin` have been set as described in above “Installation” section.

## Running Stable Diffusion in JAX on Intel GPUs

### Install Required Packages
```bash
pip install flax==0.6.6 transformers==4.27.4 diffusers==0.16.1 datasets==2.12.0
sudo apt install numactl
```
### Run Stable Diffusion Inference 

Image generation with Stable Diffusion is used for a wide range of use cases, including content creation, product design, gaming, architecture, etc. We provide the code file `jax_sd.py` below that you can copy and execute directly. The script is based on the official guide [Stable Diffusion in JAX / Flax](https://huggingface.co/blog/stable_diffusion_jax). It generates images based on a text prompt, such as `Wildlife photography of a cute bunny in the wood, particles, soft-focus, DSLR`. For benchmarking purpose, we do warmup for one-time costs for model creation and JIT compilation. Then we execute actual inference with 10 iterations and print out the average latency, for 1 image with 512 x 512 pixels generated, in a step count of 20 for each iteration. The hyper-parameters are listed in Table 1: the maximum sequence length for the prompt is set as default 77 with 768 embedding dimensions each, the guidance scale is 7.5, and the scheduler is Multistep DPM-Solver (Fast Solver for Guided Sampling of Diffusion Probabilistic Models, [multistep-dpm-solver API in Diffusers](https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver)).

<p align="center">Table 1: Text-to-images Generation Configurations in Stable Diffusion</p>
<table align="center"><tbody>
<tr>
   <td align="center">Model</td>
   <td align="center"><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">CompVis/stable-diffusion-v1-4</a></td>
</tr>
<tr>
   <td align="center">Image Resolution</td>
   <td align="center">512x512</td>
</tr>
<tr>
   <td align="center">Batch Size</td>
   <td align="center">1</td>
</tr>
<tr>
   <td align="center">Inference Iterations</td>
   <td align="center">10</td>
</tr>
<tr>
   <td align="center">Steps Count in Each Iteration</td>
   <td align="center">20</td>
</tr>
<tr>
   <td align="center">Maximum Sequence Length for Prompt</td>
   <td align="center">77</td>
</tr>
<tr>
   <td align="center">Text Embedding Dimensions</td>
   <td align="center">768</td>
</tr>
<tr>
   <td align="center">Guidance_scale</td>
   <td align="center">7.5</td>
</tr>
<tr>
   <td align="center">Scheduler</td>
   <td align="center">DPMSolverMultistepScheduler</td>
</tr>
</tbody></table>

The generated image `img.png` is saved in the current directory on the last run. 

#### Script: jax_sd.py
```python
import jax
import sys
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline, FlaxDPMSolverMultistepScheduler
import time
from PIL import Image

# Change this prompt to describe the image you’d like to generate
prompt = "Wildlife photography of a cute bunny in the wood, particles, soft-focus, DSLR"

scheduler, scheduler_state = FlaxDPMSolverMultistepScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, revision="bf16", dtype=jax.numpy.bfloat16)
params["scheduler"] = scheduler_state

prng_seed = jax.random.PRNGKey(0)

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

def elapsed_time(nb_pass=10, num_inference_steps=20):
    # warmup
    images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
    start = time.time()
    for _ in range(nb_pass):
        images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
    end = time.time()
    return (end - start) / nb_pass, images

latency, images = elapsed_time(nb_pass=10, num_inference_steps=20)
print("Latency per image is: {:.3f}s".format(latency), file=sys.stderr)
images = images.reshape((images.shape[0],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)
images[0].save("img.png")
```
#### Execute the Benchmark on Intel GPUs

Set up the environment variable of [affinity mask](https://spec.oneapi.io/level-zero/latest/core/PROG.html#affinity-mask) to utilize 1 stack of the GPU (only for 2-stacks GPU, such as Max 1550 GPU) and use `numactl` to bind the process with GPU affinity NUMA node.
```bash
export ZE_AFFINITY_MASK=0.0
numactl -N 0 -m 0 python jax_sd.py
```

## Performance Data

Based on the above steps, we measured and collected the Stable Diffusion performance data as demonstrated in Table 2 on 2 SKUs of [Intel® Data Center GPU Max Series](https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/max-series.html), Max 1550 GPU (600W OAM) and Max 1100 GPU (300W PCIe), respectively. Check out the [Intel® Data Center GPU Max Series Product Brief](https://www.intel.com/content/www/us/en/products/docs/discrete-gpus/data-center-gpu/max-series-gpu-product-brief.html) for details. On both GPUs, we could generate attractive images in less than 1 second with 20 steps! Figure 3 shows the images generated on Intel® Data Center GPU Max Series. 

These benchmarks could be re-run with the above procedures or referring to [Stable Diffusion example](https://github.com/intel/intel-extension-for-openxla/tree/main/example/stable_diffusion) to reproduce the results.

<p align="center">Table 2: Stable Diffusion Inference Performance on Intel® Data Center GPU Max Series</p>
<table align="center"><tbody>
<tr>
   <td align="center">Model</td>
   <td align="center">Precision</td>
   <td align="center">Batch Size</td>
   <td align="center">Diffusion Steps</td>
   <td align="center">Latency (s) on 1 stack of Max 1550 GPU (600W OAM)</td>
   <td align="center">Latency (s) on Single Max 1100 GPU (300W PCIe)</td>
</tr>
<tr>
   <td rowspan="2" align="center">CompVis/stable-diffusion-v1-4</td>
   <td rowspan="2" align="center">BF16</td>
   <td rowspan="2" align="center">1</td>
   <td align="center">20</td>
   <td align="center">0.79</td>
   <td align="center">0.92</td>
</tr>
<tr>
   <td align="center">50</td>
   <td align="center">1.84</td>
   <td align="center">2.15</td>
</tr>
</tbody></table>

<br><div align=center><img src="/content/dam/developer/articles/technical/accelerate-stable-diffusion-on-intel-gpus-with-intel-extension-for-openxla/Generated-images-via-Intel-Data-Center-GPU-Max-Series.png"></div>

<p align="center">Figure 3: AI Generated Contents via Intel® Data Center GPU Max Series</p>

## Summary and Future Work

Intel® Extension for OpenXLA\* leverages the PJRT interface, which simplifies ML hardware and framework integration with a unified API. It enables the Intel GPU backend for diverse AI frameworks (JAX is available, while TensorFlow and PyTorch via PyTorch-XLA are on the way). With the optimization in Intel® Extension for OpenXLA\*, JAX Stable Diffusion with BF16 archives 0.79 seconds per image latency on Intel® Data Center GPU Max 1550 and 0.92 seconds per image latency on Intel® Data Center GPU Max 1100. 

As a next step, Intel will continue working with Google to adopt the NextPluggableDevice API (see [RFC](https://docs.google.com/document/d/1S1pdXUUNYYr5cpf9F2ovGoP-Gnpxb-698tdDqaHZTok/edit?resourcekey=0-sqVzq9GnBLREq1tkzezzIQ) for details) to implement non-XLA ops on Intel GPUs to support all TensorFlow models. When available, TensorFlow support for Intel® Extension for OpenXLA\* on Intel GPUs will be in the [Intel® Extension for TensorFlow\* GitHub\* repository](https://github.com/intel/intel-extension-for-tensorflow/). 

## Resources

[Intel® Extension for OpenXLA\* GitHub\* repository](https://github.com/intel/intel-extension-for-openxla)

[More Examples Running on Intel GPUs with Intel® Extension for OpenXLA\*](https://github.com/intel/intel-extension-for-openxla/tree/main/example)

[Stable Diffusion in JAX / Flax](https://huggingface.co/blog/stable_diffusion_jax)

[Accelerate JAX models on Intel GPUs via PJRT](https://opensource.googleblog.com/2023/06/accelerate-jax-models-on-intel-gpus-via-pjrt.html)

[OpenXLA Support on GPU via PJRT](https://github.com/intel/intel-extension-for-tensorflow/blob/main/docs/guide/OpenXLA_Support_on_GPU.md)

## Acknowledgement

We would like to thank Yiqiang Li, Zhoulong Jiang, Guizi Li, Yang Sheng, River Liu from the Intel® Extension for TensorFlow* development team, Ying Hu, Kai Wang, Jianyu Zhang, Huiyan Cao, Feng Ding, Zhaoqiong Zheng, Xigui Wang, etc. from AI support team, and Zhen Han from Linux Engineering & AI enabling team for their contributions to Intel® Extension for OpenXLA*. We also offer special thanks to Sophie Chen, Eric Lin, and Jian Hui Li for their technical discussions and insights, and to collaborators from Google for their professional support and guidance. Finally, we would like to extend our gratitude to Wei Li and Fan Zhao for their great support.

## Benchmarking Hardware and Software Configuration

Measured on June 21, 2023
* Hardware configuration for Intel® Data Center GPU Max 1550: 128 Xe®-Cores in total 2 stacks, 64 Xe®-cores in 1 stack are used, ECC ON, Intel® ArcherCity server platform, 1-socket 52-cores Intel® Xeon® Platinum 8469 CPU@2.00GHz, 1 x 64 GB DDR5 4800 memory, 1 x 931.5G SanDisk SDSSDA-1700 disk, operating system: Ubuntu\* 22.04.2 LTS, 5.15.0-64-generic kernel, using Intel® Xe® Matrix Extensions (Intel® XMX) BF16 with Intel® oneAPI Deep Neural Network Library (oneDNN) v3.2 optimized kernels integrated into Intel® Extension for OpenXLA v0.1.0, JAX v0.4.7 and jaxlib v0.4.7, flax v0.6.6, diffusers v0.16.1, Intel® 627.7 GPU driver, and Intel® oneAPI Base Toolkit 2023.1. Performance varies by use, configuration and other factors. Learn more at [www.Intel.com/PerformanceIndex](www.Intel.com/PerformanceIndex).
* Hardware configuration for Intel® Data Center GPU Max 1100: 56 Xe®-Cores in total 1 stack, ECC ON, Supermicro® SYS-420GP-TNR server platform, 2-sockets 32-cores Intel® Xeon® Platinum 8352Y CPU@2.20GHz, 4 x 64 GB DDR4 3200 memory, 1 x 894.3G Samsung® MZ1LB960HAJQ-00007 disk, operating system: SUSE\* Linux Enterprise Server 15 SP4, 5.14.21-150400.24.46-default kernel, using Intel® Xe® Matrix Extensions (Intel® XMX) BF16 with Intel® oneAPI Deep Neural Network Library (oneDNN) v3.2 optimized kernels integrated into Intel® Extension for OpenXLA v0.1.0, JAX v0.4.7 and jaxlib v0.4.7, flax v0.6.6, diffusers v0.16.1, Intel® 627.7 GPU driver, and Intel® oneAPI Base Toolkit 2023.1. Performance varies by use, configuration and other factors. Learn more at [www.Intel.com/PerformanceIndex](www.Intel.com/PerformanceIndex).
