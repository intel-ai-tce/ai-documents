# Setup Instructions

Please make sure to follow [Driver Installation](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) to install Gaudi driver on the system.
We suggest to use pytorch docker image to run below examples.

To use dockerfile provided for the sample, please follow [Docker Installation](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html) to setup habana runtime for Docker images.  
The docker image helps users to setup pytorch software and packages to run the samples. Users still need to install required packages like deepspeed to run the samples.  

### Docker Run
After docker build, users could follow below command to run and docker instance and users will be in the docker instance under text-generation folder.
```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none   --cap-add=ALL --privileged=true  --net=host --ipc=host  -v "$PWD":/workspace --workdir  /workspace  vault.habana.ai/gaudi-docker/1.19.1/ubuntu24.04/habanalabs/pytorch-installer-2.5.1:latest
```
> [!NOTE]
> The Huggingface model file size might be large, so we recommend to use an external disk as Huggingface hub folder. \
> Please export HF_HOME environment variable to your external disk and then export the mount point into docker instance. \
> ex: "-e HF_HOME=/mnt/huggingface -v /mnt:/mnt"

### Install required packages inside docker
First, you should install the requirements:
```bash
pip install -r requirements.txt
```

For `run_lm_eval.py`:
```bash
pip install -r requirements_lm_eval.txt
```

Then, if you plan to use [DeepSpeed-inference](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Inference_Using_DeepSpeed.html) (e.g. to use BLOOM/BLOOMZ), you should install DeepSpeed as follows:
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
```
# Tensor quantization statistics measurement

## Llama2
Here is an example to measure the tensor quantization statistics on LLama2:

Users could export different values to below enivironment variables to change parameters for tensor quantization statisics  
| Environment Variable | Values |
|------------------|------------|
| model_name | meta-llama/Llama-2-70b-hf ,  meta-llama/Llama-2-7b-hf |

Here is an example to run llama2-70b
```bash
export model_name=meta-llama/Llama-2-70b-hf
```

```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python ../gaudi_spawn.py \
--use_deepspeed --world_size 8 run_lm_eval.py \
-o acc_${model_name}_bs1_measure.txt \
--model_name_or_path ${model_name} \
--attn_softmax_bf16 \
--use_hpu_graphs \
--trim_logits \
--use_kv_cache \
--bucket_size=128 \
--bucket_internal \
--use_flash_attention \
--flash_attention_recompute \
--bf16 \
--batch_size 1
```

## Llama3
Here is an example to measure the tensor quantization statistics on Llama3 with 8 cards:
> Please note that Llama3-405B requires minimum 16 cards Gaudi2 and 8 cards Gaudi3.

Users could export different values to below enivironment variables to change parameters for tensor quantization statisics  
| Environment Variable | Values |
|------------------|------------|
| model_name | meta-llama/Llama-3.1-405B-Instruct , meta-llama/Llama-3.1-70B-Instruct, and meta-llama/Llama-3.1-8B-Instruct |

Here is an example to run llama3-405b
```bash
export model_name=meta-llama/Llama-3.1-405B-Instruct
```

```bash
QUANT_CONFIG=./quantization_config/maxabs_measure_include_outputs.json python ../gaudi_spawn.py \
--use_deepspeed --world_size 8 run_lm_eval.py \
-o acc_${model_name}_bs1_quant.txt \
--model_name_or_path ${model_name} \
--use_hpu_graphs \
--use_kv_cache \
--trim_logits \
--batch_size 1 \
--bf16 \
--reuse_cache \
--use_flash_attention \
--flash_attention_recompute \
--flash_attention_causal_mask
```

# Quantize and run the fp8 model

Here is an example to quantize the model based on previous measurements for LLama2 or 3 models:

Users could export different values to below enivironment variables to change parameters for benchmarking
| Environment Variable | Values |
|------------------|------------|
| model_name | meta-llama/Llama-2-70b-hf , meta-llama/Llama-2-7b-hf, meta-llama/Llama-3.1-405B-Instruct , meta-llama/Llama-3.1-70B-Instruct, and meta-llama/Llama-3.1-8B-Instruct |
| input_len | 128, 2048, and etc |
| output_len | 128, 2048, and etc |
| batch_size | 350, 1512, 1750, and etc |

Here is an example to run llama2-70b with input tokens lenght=128, output tokens length=128 and batch size = 1750 
```bash
export model_name=meta-llama/Llama-2-70b-hf
export input_len=128
export output_len=128
export batch_size=1750
```
After setting the environment variables, users could run the fp8 model by below command.  
```bash
QUANT_CONFIG=./quantization_config/maxabs_quant.json python ../gaudi_spawn.py \
--use_deepspeed --world_size 8 run_generation.py \
--model_name_or_path ${model_name} \
--attn_softmax_bf16 \
--use_hpu_graphs \
--limit_hpu_graphs \
--trim_logits \
--use_kv_cache \
--reuse_cache \
--use_flash_attention \
--flash_attention_recompute \
--flash_attention_causal_mask  \
--bucket_size=128 \
--bucket_internal \
--bf16 \
--batch_size ${batch_size} \
--max_new_tokens ${output_len} \
--max_input_tokens ${input_len} \
--book_source \
--warmup 2
```


