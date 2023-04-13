
## Overview
[Mixed precision](https://www.tensorflow.org/guide/mixed_precision) is the use of both 16-bit and 32-bit floating-point types in a model during training and inference to make it run faster and use less memory. Official Tensorflow* supports this feature from TensorFlow 2.9 with all Intel® optimizations enabled by default. Intel® 4th Gen Xeon processor codenamed Sapphie Rapids(SPR) is the first Intel® processor to support Advanced Matrix Extensions(AMX) instructions, which help to accelerate matrix heavy workflows such as machine learning alongside also having BFloat16 support. Previous hardwares can be used to test functionality, but SPR provides the best performance gains. 

Using one of the methods described here will convert a model written in FP32 data type to operate in BFloat16 data type. To be precise, it scans the data-flow graph corresponding to the model, and looks for nodes in the graph (also called as operators) that can operate in BFloat16 type, and inserts FP32ToBFloat16 and vice-versa Cast nodes in the graph appropriately.

For example: Consider a simple neural network with a typical pattern of Conv2D -> BiasAdd -> Relu with the default TensorFlow datatype FP32. TensorFlow* data-flow graph corresponding to this model looks like below.

![amp-mkl](/content/dam/develop/external/us/en/images/automixedprecisionmkl-1.png)

The data-flow graph after porting the model to BFloat16 type looks like below.

![amp-mkl2](/content/dam/develop/external/us/en/images/automixedprecisionmkl-2.png)

Notice that 2 operators, Conv2D+BiasAdd and ReLU in the graph are automatically converted to operate in BFloat16 type. Also note that appropriate Cast nodes are inserted in the graph to convert TensorFlow* tensors from FP32 type to BFloat16 type and vice-versa.

Here are the options to enable BF16 mixed precision in TensorFlow

1. Keras mixed precision API
2. Mixed Precision for Tensorflow Hub models
2. Legacy AutoMixedPrecision grappler pass
    
This guide describes the options in details.

## I. Keras mixed precision API
### Introduction
This session describes how to use the Keras mixed precision API to speed up your models by using a [Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet/) instead.

### Getting started with a FP32 Model
By using an existing sample, users can easily understand how to change their own Keras* model.

First create an environment and install Tensorflow*
```bash
conda create -n tf_latest python=3.8 -c intel 
source activate tf_latest 
(tf_latest) pip install tensorflow
```

Next, get the sample model using the command below and then using python train the model.
Note: For quick experimentation, one can change epoch value from 15 to 2 in line#67 of mnist_convert.py
```bash
wget https://raw.githubusercontent.com/keras-team/keras-io/master/examples/vision/mnist_convnet.py
python mnist_convnet.py
```
### Steps to Train a BFloat16 Model
To use bfloat16 as compute data type in the above model, users only need to add the below two lines around line#16 of mnist_convert.py.

```python
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_bfloat16')
```
Users can use oneDNN verbose logs to verify if this modified mnist_convnet indeed uses bfloat16 data type for computation. Using ONEDNN_VERBOSE=1 environment variable as shown below, users will get a oneDNN verbose log file "dnnl_log.csv".

```bash
ONEDNN_VERBOSE=1 python mnist_convnet.py > dnnl_log.csv
```

With **Analyze Verbose Logs** section in this [Tutorial](https://github.com/oneapi-src/oneAPI-samples/blob/master/Libraries/oneDNN/tutorials/profiling/README.md), users can parse the dnnl_log.csv and get oneDNN JIT kernel breakdown.

Here is the result from modified mnist_convnet.py on Sapphie Rapids(SPR) with BF16 AMX enabled.

From below result, avx512_core_bf16 jit kernel indeed was used. Moreover, avx512_core_amx_bf16 jit kernel which supports AMX feature was also used and it took ~85% of total elapsed time. 

```bash
jit kernel                             elapsed time
brgemm:avx512_core_amx_bf16              199.422110
brg:avx512_core_amx_bf16                2467.363857
brgconv:avx512_core_bf16                2573.850987
jit:avx512_core                         6672.159937
jit:avx512_core_bf16                   11135.476744
brgconv:avx512_core_amx_bf16           12327.448310
jit:uni                                33479.923155
brgconv_bwd_w:avx512_core_amx_bf16    150781.885700
```

> Please use Python 3.8 or above for the best performance on Sapphire Rapids Xeon Scalable Processors.

## II. Mixed Precision for Tensorflow Hub models 
### Introduction
This session describes how to enable the auto-mixed precision for [TensorFlow Hub](https://www.tensorflow.org/hub) models using the tf.config API. Enabling this API will automatically convert the pre-trained model to use the bfloat16 datatype for computation resulting in an increased training throughput on the latest Intel® Xeon® scalable processor. These instructions works for inference or fine tuning usecase.

### Steps to do Transfer Learning with Mixed Precision on a Saved Model
Please refer to [this Transfer Learning sample](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Features-and-Functionality/IntelTensorFlow_Enabling_Auto_Mixed_Precision_for_TransferLearning) which uses a headless ResNet50v1.5 pretrained model from [TensorFlow Hub](https://www.tensorflow.org/hub) with ImageNet dataset.

By applying below experimental option "auto_mixed_precision_onednn_bfloat16" via tf.config API, auto mixed precision with bfloat16 will be enabled on the saved model.

```python
tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16':True})
```

### Steps to enable Bfloat16 Inference for a Saved Model
Same config as earlier can be used.
```python
tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16':True})
```
 

## III. Legacy Auto Mixed Precision (AMP) with Bfloat16
Auto Mixed Precision is a grappler pass that automatically converts a model written in FP32 data type to operate in BFloat16 data type. It mainly supports TF v1 style models that use a session to run the model. We will demonstrate this with examples illustrating:

- How to convert the graph to BFloat16 on-the-fly as you train the model
- How to convert a pre-trained fp32 model to BFloat16

Let’s use a simple neural network with a typical pattern of Conv2D -> BiasAdd -> ReLU. Inputs x and w of Conv2D and input b of bias_add are TensorFlow* Variables with default FP32 data type, so this neural network model operates completely in FP32 data type. TensorFlow* data-flow graph corresponding to this model looks like below.

### Steps to set up the runtime environment
Before running the examples, setup runtime environement with steps below to create an environment and install Tensorflow*, as done earlier.
```bash
conda create -n tf_latest python=3.8 -c intel 
source activate tf_latest 
(tf_latest) pip install tensorflow
```

### Enabling BFloat16 using AMP in a Model
Following lines of Python* code will enable BFloat16 data type using Auto Mixed Precision when using grappler via Tensorflow v1. This setting enables BF16 for training AND inference usecase.
```python
graph_options=tf.compat.v1.GraphOptions( 
        rewrite_options=rewriter_config_pb2.RewriterConfig( 
            auto_mixed_precision_onednn_bfloat16=rewriter_config_pb2.RewriterConfig.ON)) 
```
> Note: auto_mixed_precision_onednn_bfloat16 was known as auto_mixed_precision_mkl since Tensorflow 2.3 till Tensorflow 2.9. However, auto_mixed_precision_mkl is deprecated and will be removed in future.

Auto Mixed Precision grappler pass is disabled by default and can be controlled using RewriterConfig proto of GraphOptions proto. To enable use rewriter_config_pb2.RewriterConfig.ON and to disable use rewriter_config_pb2.RewriterConfig.OFF (default).

Below is the complete code for a neural network model ported to BFloat16 data type using Auto Mixed Precision.

```python
import tensorflow as tf 
from tensorflow.core.protobuf import rewriter_config_pb2 

tf.compat.v1.disable_eager_execution()

def conv2d(x, w, b, strides=1): 
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME') 
    x = tf.nn.bias_add(x, b) 
    return tf.nn.relu(x)  

X = tf.Variable(tf.compat.v1.random_normal([784])) 
W = tf.Variable(tf.compat.v1.random_normal([5, 5, 1, 32])) 
B = tf.Variable(tf.compat.v1.random_normal([32])) 
x = tf.reshape(X, shape=[-1, 28, 28, 1])  

# Note: Auto Mixed Precision pass is enabled here to run on Xeon CPUs with BFloat16
graph_options=tf.compat.v1.GraphOptions( 
        rewrite_options=rewriter_config_pb2.RewriterConfig( 
            auto_mixed_precision_onednn_bfloat16=rewriter_config_pb2.RewriterConfig.ON))  

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto( 
        graph_options=graph_options)) as sess: 
    sess.run(tf.compat.v1.global_variables_initializer()) 
    sess.run([conv2d(x, W, B)]) 
```

Notice that graph_options variable created by turning Auto Mixed Precision ON is passed to ConfigProto that is eventually passed to tf.Session API.

Create and run the Python* file with the above code **conv2D_bf16.py**

```bash
python conv2D_bf16.py 
```

**Console output**
```bash
2022-08-19 00:06:08.735706: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:2303] Running auto_mixed_precision_onednn_bfloat16 graph optimizer
2022-08-19 00:06:08.735941: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:1488] No allowlist ops found, nothing to do
2022-08-19 00:06:08.750799: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:2303] Running auto_mixed_precision_onednn_bfloat16 graph optimizer
2022-08-19 00:06:08.751058: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:2240] Converted 3/11 nodes to bfloat16 precision using 0 cast(s) to bfloat16 (excluding Const and Variable casts)
```
> _No allowlist ops found typically refers to the data session._
> One can also use ONEDNN_VERBOSE logs as discussed earlier to verify AMX BF16 instructions.

### Controlling Ops to be converted to BFloat16 type Automatically
An important point to note is that not all of TensorFlow’s* operators for CPU backend support BFloat16 type - this could be because either the support is missing (and is a WIP) or that the BFloat16 version of an operator may not offer much performance improvement over the FP32 version.

Furthermore, BFloat16 type for certain operators could lead to numerical instability of a neural network model. So we categorize TensorFlow* operators that are supported by MKL backend in BFloat16 type into 1) if they are always numerically stable, and 2) if they are always numerically unstable, and 3) if their stability could depend on the context. Auto Mixed Precision pass uses a specific Allow, Deny and Infer list of operators respectively to capture these operators. The exact lists could be found in auto_mixed_precision_lists.h file in TensorFlow* github repository.

We would like to mention that the default values of these lists already capture the most common BFloat16 usage models and to also ensure numerical stability of the model. There is, however, a way to add or remove operators from any of these lists as by setting environment variables that control these lists. For instance, executing

```bash
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_ADD=Conv2D 
```
before running the model would add Conv2D operator to Deny list. While, executing
```bash
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_REMOVE=Conv2D 
```
before running the model would remove Conv2D from Allow list. And executing both of these commands before running a model would move Conv2D from Allow list to Deny list.

In general, the template corresponding to the names of the environment variables controlling these lists is:
```bash
TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_${LIST}_${OP}=operator
```
where ${LIST} would be any of {ALLOW, DENY, INFER}, and ${OP} would be any of {ADD, REMOVE}.

To test this feature of adding an op into denylist and removing from the allowlist, run the code sample conv2D_bf16.py by enabling the environment variables with conv2d ops
```bash
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_ADD=Conv2D 
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_REMOVE=Conv2D  
python conv2D_bf16.py 
```
As you can see from the console output and comparing with the Step 1, conv2D ops have been removed from the allowlist and added to the denylist, thus controlling the node from converting into BFloat16 equivalent.
```bash
2022-08-19 00:08:41.975709: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:2303] Running auto_mixed_precision_onednn_bfloat16 graph optimizer
2022-08-19 00:08:41.975944: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:1488] No allowlist ops found, nothing to do
2022-08-19 00:08:41.990983: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:2303] Running auto_mixed_precision_onednn_bfloat16 graph optimizer
2022-08-19 00:08:41.991196: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:1488] No allowlist ops found, nothing to do
```
 
