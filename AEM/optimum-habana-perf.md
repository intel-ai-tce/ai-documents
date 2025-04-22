# Setup Instructions

Please make sure to follow [Driver Installation](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) to install the Gaudi driver on the system.  
It is recommended to use the Optimum-Habana fp8 Benchmark [Dockerfile](https://github.com/HabanaAI/Gaudi-tutorials/blob/main/PyTorch/Hugging_Face_pipelines/Benchmarking_on_Optimum-habana_with_fp8/Dockerfile) to run the examples below.

To use the provided Dockerfile for the sample, follow the [Docker Installation guide](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html) to setup the Habana runtime for Docker images.  

## Build and Run the Benchmark Docker instance 

### Get Dockerfile and Benchmark scripts
First, get the Dockerfile and Benchmark scripts from Gaudi-Tutorial GitHub Repositroy following below command.  

```bash
git clone https://github.com/HabanaAI/Gaudi-tutorials.git
cd Gaudi-tutorials/PyTorch/Hugging_Face_pipelines/Benchmarking_on_Optimum-habana_with_fp8
```
### Docker Build
To build the image from the Dockerfile, please follow below command to build the optimum-habana-text-gen image.
```bash
docker build --no-cache -t optimum-habana-text-gen:latest --build-arg https_proxy=$https_proxy --build-arg http_proxy=$http_proxy -f Dockerfile .
```
### Docker Run
After docker build, users could follow below command to run and docker instance and users will be in the docker instance under text-generation folder.
```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none   --cap-add=ALL --privileged=true  --net=host --ipc=host optimum-habana-text-gen:latest
```
> [!NOTE]
> The Huggingface model file size might be large, so we recommend to use an external disk as Huggingface hub folder. \
> Please export HF_HOME environment variable to your external disk and then export the mount point into docker instance. \
> ex: "-e HF_HOME=/mnt/huggingface -v /mnt:/mnt"

# Run Benchmark with Benchmark.py
Benchmark script will run all the models with different input len, output len and batch size and generate a report to compare all published numbers in [Gaudi Model Performance](https://www.intel.com/content/www/us/en/developer/platform/gaudi/model-performance.html).  

## Gaudi3
Different json file are provided for different Gaudi Software version like 1.19 and 1.20 on Gaudi3.
To do benchmarking on a machine with 8 Gaudi3 cards, just run the below command inside the docker instance. 
```bash
python3 Benchmark.py
```
## Gaudi2
To do benchmarking on a machine with 8 Gaudi2 cards, just run the below command instead inside the docker instance. 
```bash
GAUDI_VER=2 python3 Benchmark.py
```
## HTML Report
A html report will be generated under a folder with timestamp.  
The [html report](https://github.com/HabanaAI/Gaudi-tutorials/tree/main/PyTorch/Hugging_Face_pipelines/Benchmarking_on_Optimum-habana_with_fp8#html-report) also has a perf_ratio column to compare the measure numbers with reference perf numbers. 

# Run Benchmark without Benchmark.py
<details>
<summary> Run instructions without using the Benchmark script  </summary>
  
## Tensor quantization statisics measurement
This step needs to be completed only once for each model with the corresponding world size values.  
The hqt_output generated after this step will be used for the FP8 run.  
If changing models for the FP8 run, repeat this step to obtain the relevant hqt_output.  

Here is an example to measure the tensor quantization statistics for LLama2 or 3 models:  
{:.note}Please note that Llama3-405B requires a minimum of 8 Gaudi3 cards.

Export different values to the following environment variables to change parameters for tensor quantization statistics:  
| Environment Variable | Values |
|------------------|------------|
| model_name | meta-llama/Llama-2-70b-hf,  meta-llama/Llama-2-7b-hf, meta-llama/Llama-3.1-405B-Instruct, meta-llama/Llama-3.1-70B-Instruct, meta-llama/Llama-3.3-70B-Instruct, and meta-llama/Llama-3.1-8B-Instruct |
| world_size | 1, 2, 8 |

```bash
export model_name=meta-llama/Llama-2-70b-hf
export world_size=2
```

```bash
HF_DATASETS_TRUST_REMOTE_CODE=true QUANT_CONFIG=./quantization_config/maxabs_measure.json TQDM_DISABLE=1 python3 ../gaudi_spawn.py \
--use_deepspeed --world_size ${world_size} run_lm_eval.py \
-o acc_llama_quant.json \
--model_name_or_path ${model_name} \
--warmup 0 \
--use_hpu_graphs \
--use_kv_cache \
--trim_logits \
--batch_size 1 \
--bucket_size=128 \
--bucket_internal \
--trust_remote_code \
--tasks hellaswag lambada_openai piqa winogrande \
--bf16 \
--attn_softmax_bf16 \
--use_flash_attention \
--flash_attention_recompute \
--flash_attention_causal_mask
```

## Quantize and run the fp8 model

Here is an example to quantize the model based on previous measurements for LLama2 or 3 models:

Export different values to the following environment variables to change parameters for tensor quantization statistics:  
| Environment Variable | Values |
|------------------|------------|
| model_name | meta-llama/Llama-2-70b-hf, meta-llama/Llama-2-7b-hf, meta-llama/Llama-3.1-405B-Instruct, meta-llama/Llama-3.1-70B-Instruct, meta-llama/Llama-3.3-70B-Instruct, and meta-llama/Llama-3.1-8B-Instruct |
| input_len | 128, 2048, and etc |
| output_len | 128, 2048, and etc |
| batch_size | 350, 1512, 1750, and etc |
| world_size | 1, 2, 8 |

{:.note}Please note that Llama3-405B requires a minimum of 8 Gaudi3 cards.

Here is an example to run llama2-70b with input tokens length=128, output tokens length=128 and batch size = 1750 
```bash
export model_name=meta-llama/Llama-2-70b-hf
export input_len=128
export output_len=128
export batch_size=1750
export world_size=2
```
After setting the environment variables, run the FP8 model using the following command:  
```bash
HF_DATASETS_TRUST_REMOTE_CODE=true QUANT_CONFIG=./quantization_config/maxabs_quant.json TQDM_DISABLE=1 python3 ../gaudi_spawn.py \
--use_deepspeed --world_size ${world_size} run_generation.py \
--model_name_or_path ${model_name} \
--attn_softmax_bf16 \
--use_hpu_graphs \
--limit_hpu_graphs \
--trim_logits \
--use_kv_cache \
--use_flash_attention \
--flash_attention_recompute \
--flash_attention_causal_mask  \
--bucket_size=128 \
--bucket_internal \
--attn_batch_split 2  \
--bf16 \
--batch_size ${batch_size} \
--max_new_tokens ${output_len} \
--max_input_tokens ${input_len} \
--warmup 2
```
{:.note}Please note that Llama3-405B requires --book\_source additionally to achieve better performance. Llama3.3-70B model also doesn't require the "--attn\_batch\_split 2" argument.
</details>

