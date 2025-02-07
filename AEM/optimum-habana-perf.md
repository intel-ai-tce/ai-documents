# Setup Instructions

Please make sure to follow [Driver Installation](https://docs.habana.ai/en/latest/Installation_Guide/Driver_Installation.html) to install the Gaudi driver on the system.  
It is recommended to use the PyTorch Docker image to run the examples below.

To use the provided Dockerfile for the sample, follow the [Docker Installation guide](https://docs.habana.ai/en/latest/Installation_Guide/Additional_Installation/Docker_Installation.html) to setup the Habana runtime for Docker images.  
The Docker image assists in setting up the PyTorch software and packages to run the samples. However, installing additional required packages like DeepSpeed is still necessary to run the samples. 

### Get examples from optimum-habana github repository
To benchmark Llama2 and Llama3 models, obtain optimum-habana from the GitHub repository using the following command.
```bash
git clone -b v1.15.0 https://github.com/huggingface/optimum-habana.git
cd optimum-habana/examples/text-generation
```

### Docker Run
After building the Docker image, run the following command to start a Docker instance, which will open in the text-generation folder inside the docker instance.
```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none   --cap-add=ALL --privileged=true  --net=host --ipc=host  -v "$PWD/../../":/workspace --workdir  /workspace/examples/text-generation  vault.habana.ai/gaudi-docker/1.19.0/ubuntu24.04/habanalabs/pytorch-installer-2.5.1:latest
```
>**NOTE:**
> The Huggingface model file size might be large, so it is recommended to use an external disk as the Huggingface hub folder. \
> Export the HF_HOME environment variable to the external disk and then export the mount point into the Docker instance. \
> ex: "-e HF_HOME=/mnt/huggingface -v /mnt:/mnt"

### Install required packages inside docker
First, install the optimum-habana:
```bash
pip install --upgrade-strategy eager optimum[habana]
```

Second, install the requirements:
```bash
pip install -r requirements.txt
```

For `run_lm_eval.py`:
```bash
pip install -r requirements_lm_eval.txt
```

Then, to use [DeepSpeed-inference](https://docs.habana.ai/en/latest/PyTorch/DeepSpeed/Inference_Using_DeepSpeed.html), install DeepSpeed as follows: 
```bash
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
```


# Tensor quantization statisics measurement
This step needs to be completed only once for each model with the corresponding world size values.  
The hqt_output generated after this step will be used for the FP8 run.  
If changing models for the FP8 run, repeat this step to obtain the relevant hqt_output.  
### Llama2
Here is an example to measure the tensor quantization statistics on LLama2:

Export different values to the following environment variables to change parameters for tensor quantization statistics:    
| Environment Variable | Values |
|------------------|------------|
| model_name | meta-llama/Llama-2-70b-hf,  meta-llama/Llama-2-7b-hf |
| world_size | 1, 2, 8 |

```bash
export model_name=meta-llama/Llama-2-70b-hf
export world_size=2
```

```bash
QUANT_CONFIG=./quantization_config/maxabs_measure.json python3 ../gaudi_spawn.py \
--use_deepspeed --world_size ${world_size} run_lm_eval.py \
-o acc_llama2_bs1_measure.txt \
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

### Llama3
Here is an example to measure the tensor quantization statistics on Llama3 with 8 cards:
> Please note that Llama3-405B requires a minimum of 8 Gaudi3 cards.

Export different values to the following environment variables to change parameters for tensor quantization statistics:  
| Environment Variable | Values |
|------------------|------------|
| model_name | meta-llama/Llama-3.1-405B-Instruct, meta-llama/Llama-3.1-70B-Instruct, and meta-llama/Llama-3.1-8B-Instruct |
| world_size | 8 |

```bash
export model_name=meta-llama/Llama-3.1-405B-Instruct
export world_size=8
```

```bash
QUANT_CONFIG=./quantization_config/maxabs_measure_include_outputs.json python3 ../gaudi_spawn.py \
--use_deepspeed --world_size ${world_size} run_lm_eval.py \
-o acc_llama3_bs1_quant.txt \
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

Export different values to the following environment variables to change parameters for tensor quantization statistics:  
| Environment Variable | Values |
|------------------|------------|
| model_name | meta-llama/Llama-2-70b-hf, meta-llama/Llama-2-7b-hf, meta-llama/Llama-3.1-405B-Instruct, meta-llama/Llama-3.1-70B-Instruct, and meta-llama/Llama-3.1-8B-Instruct |
| input_len | 128, 2048, and etc |
| output_len | 128, 2048, and etc |
| batch_size | 350, 1512, 1750, and etc |
| world_size | 1, 2, 8 |

> Please note that Llama3-405B requires a minimum of 8 Gaudi3 cards.

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
QUANT_CONFIG=./quantization_config/maxabs_quant.json python3 ../gaudi_spawn.py \
--use_deepspeed --world_size ${world_size} run_generation.py \
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


