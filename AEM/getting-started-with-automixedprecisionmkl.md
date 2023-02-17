​
Overview
Mixed precision is the use of both 16-bit and 32-bit floating-point types in a model during training to make it run faster and use less memory.

There are two options to enable BF16 mixed precision in Intel Optimized TensorFlow.

Keras mixed precision API
AutoMixedPrecision oneDNN BFloat16 grappler pass through low level session configuration
This guide describes how to enable BF16 mixed precision via those two options.

 Keras mixed precision API
Introduction
This session describes how to use the Keras mixed precision API to speed up your models by using a Simple MNIST convnet instead.

Official Tensorflow* supports this feature from TensorFlow 2.9. And starting from TensorFlow 2.4, Intel Optimized TensorFlow supports the experimental Keras mixed precision API.

Steps to prepare a FP32 Model
By using an existing sample, users can easily understand how to change their own Keras* model.

First, get the sample by using the command below:

wget https://raw.githubusercontent.com/keras-team/keras-io/master/examples/vision/mnist_convnet.py

We don't need to train model with 15 epochs, so change epoch value from 15 to 2 in line#67 of mnist_convert.py:

epoch = 2

Then, use the command below to train the convnet model.

python mnist_convnet.py

Steps to Train a BFloat16 Model
To port high-level Keras* API convnet sample, users only need to add the below three lines around line#16 of mnist_convert.py.

The model will use bfloat16 as its compute data type.

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_global_policy(policy)

Users could also use oneDNN verbose log to verify if this modified mnist_convnet indeed uses bfloat16 data type for computation.

By following below command, users will get a oneDNN verbose log file "dnnl_log.csv".

ONEDNN_VERBOSE=1 python mnist_convnet.py > dnnl_log.csv

  By following Analyze Verbose Logs section in this Tutorial, users can parse this dnnl_log.csv and get a oneDNN JIT kernel breakdown.

Here is the result from modified mnist_convnet.py on Sapphie Rapids Xeon Scalable Processors. 

Sapphire Rapids is the first Intel processors to support Advanced Matrix Extensions (AMX), which we understand to help accelerate matrix heavy workflows such as machine learning alongside also having BFloat16 support. 

From below result, avx512_core_bf16 jit kernel indeed was used. Moreover, avx512_core_amx_bf16 jit kernel which supports AMX feature was also used and it took ~85% of total elapsed time. 

jit kernel                             elapsed time
brgemm:avx512_core_amx_bf16              199.422110
brg:avx512_core_amx_bf16                2467.363857
brgconv:avx512_core_bf16                2573.850987
jit:avx512_core                         6672.159937
jit:avx512_core_bf16                   11135.476744
brgconv:avx512_core_amx_bf16           12327.448310
jit:uni                                33479.923155
brgconv_bwd_w:avx512_core_amx_bf16    150781.885700


 

Please use Python 3.8 or above for the best performance on Sapphire Rapids Xeon Scalable Processors.

Enable Mixed Precision on a Saved Model for Transfer Learning
Introduction
This session describes how to enable the auto-mixed precision using the tf.config API. Enabling this API will automatically convert the pre-trained model to use the bfloat16 datatype for computation resulting in an increased training throughput on the latest Intel® Xeon® scalable processor how to enable the auto-mixed precision using the tf.config API. Enabling this API will automatically convert the pre-trained model to use the bfloat16 datatype for computation resulting in an increased training throughput on the latest Intel® Xeon® scalable processor.

Steps to do Transfer Learning with Mixed Precision on a Saved Model
Please refer to this Transfer Learning sample which uses a headless ResNet50v1.5 pretrained model from TensorFlow Hub with ImageNet dataset.

By applying below experimental option "auto_mixed_precision_onednn_bfloat16" via tf.config API, auto mixed precision with bfloat16 will be enabled on the saved model.

tf.config.optimizer.set_experimental_options({'auto_mixed_precision_onednn_bfloat16':True})

 

Auto Mixed Precision with Bfloat16
Introduction
Auto Mixed Precision is a grappler pass that automatically converts a model written in FP32 data type to operate in BFloat16 data type. To be precise, it scans the data-flow graph corresponding to the model, and looks for nodes in the graph (also called as operators) that can operate in BFloat16 type, and inserts FP32ToBFloat16 and vice-versa Cast nodes in the graph appropriately. This feature is available in official TensorFlow* since TensorFlow* 2.9, however, it has been available in Intel* Optimized TensorFlow* since TensorFlow* 2.3. We will demonstrate this with examples illustrating:

How to convert the graph to BFloat16 on-the-fly as you train the model

How to convert a pre-trained fp32 model to BFloat16

Let’s consider a simple neural network consisting of a typical pattern of Conv2D with addition of bias and output of Conv2D clipped using ReLU. Inputs x and w of Conv2D and input b of bias_add are TensorFlow* Variables. Notice that the default type of Variable in TensorFlow* is FP32, so this neural network model operates completely in FP32 data type. TensorFlow* data-flow graph corresponding to this model looks like below.



Steps to set up the runtime environment
Before running the examples build or install the latest TensorFlow* with BFloat16 support

Follow the below instructions to simply create an environment and install Tensorflow*

conda create -n tf_latest python=3.8 -c intel 
source activate tf_latest 
(tf_latest) pip install tensorflow

 

Steps to prepare a FP32 Model
Let's create a sample workload conv2D_fp32.py with a Conv2d and Relu layers.

import tensorflow as tf 
from tensorflow.core.protobuf import rewriter_config_pb2  

# Disable Eager execution mode 
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

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto()) as sess: 
    sess.run(tf.compat.v1.global_variables_initializer()) 
    sess.run([conv2d(x, W, B)]) 

python conv2D_fp32.py 

This will train a regular fp32 conv2d graph.

Steps to Train a BFloat16 Model
Porting the above workload to BFloat16 data type using Auto Mixed Precision requires adding following lines of Python* code to this model:

graph_options=tf.compat.v1.GraphOptions( 
        rewrite_options=rewriter_config_pb2.RewriterConfig( 
            auto_mixed_precision_onednn_bfloat16=rewriter_config_pb2.RewriterConfig.ON)) 

Note: auto_mixed_precision_onednn_bfloat16 was known as auto_mixed_precision_mkl since Tensorflow 2.3 till Tensorflow 2.9. However, auto_mixed_precision_mkl is deprecated and will be removed in future.

In a nutshell, Auto Mixed Precision grappler pass can be controlled using RewriterConfig proto of GraphOptions proto. Possible values to initialize auto_mixed_precision_onednn_bloat16 are rewriter_config_pb2.RewriterConfig.ON and rewriter_config_pb2.RewriterConfig.OFF.

The default value is rewriter_config_pb2.RewriterConfig.OFF.

Add the code snippet given above just before intializing a TensorFlow* session with the graph_options to include the grappler pass.

Below is the complete code for the neural network model that is ported to BFloat16 type using Auto Mixed Precision is below.

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

graph_options=tf.compat.v1.GraphOptions( 
        rewrite_options=rewriter_config_pb2.RewriterConfig( 
            auto_mixed_precision_onednn_bfloat16=rewriter_config_pb2.RewriterConfig.ON))  

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto( 
        graph_options=graph_options)) as sess: 
    sess.run(tf.compat.v1.global_variables_initializer()) 
    sess.run([conv2d(x, W, B)]) 

Notice that graph_options variable created by turning Auto Mixed Precision ON is passed to ConfigProto that is eventually passed to tf.Session API.

Create and run the Python* file with the above code conv2D_bf16.py

python conv2D_bf16.py 

Console output

2022-08-19 00:06:08.735706: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:2303] Running auto_mixed_precision_onednn_bfloat16 graph optimizer
2022-08-19 00:06:08.735941: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:1488] No allowlist ops found, nothing to do
2022-08-19 00:06:08.750799: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:2303] Running auto_mixed_precision_onednn_bfloat16 graph optimizer
2022-08-19 00:06:08.751058: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:2240] Converted 3/11 nodes to bfloat16 precision using 0 cast(s) to bfloat16 (excluding Const and Variable casts)

The data-flow graph after porting the model to BFloat16 type looks like below.



Notice that 2 operators, Conv2D+BiasAdd and ReLU in the graph are automatically converted to operate in BFloat16 type. Also note that appropriate Cast nodes are inserted in the graph to convert TensorFlow* tensors from FP32 type to BFloat16 type and vice-versa.

 
Steps to Convert a Pretrained fp32 Model to BFloat16
In the previous section we saw how the AutoBFloat16Convertor can automatically convert certain nodes to BFloat16 while training a sample model. This section will cover how to convert a pre-trained fp32 model to BFloat16.

1. Modify the conv2D_fp32.py to save the trained model
import tensorflow as tf 
import tensorflow.python.saved_model 
from tensorflow.python.saved_model import tag_constants 
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def 

# Disable Eager execution mode 
tf.compat.v1.disable_eager_execution() 

def conv2d(x, w, b, strides=1): 
    # Conv2D wrapper, with bias and relu activation 
    x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME',name="myConv") 
    x = tf.nn.bias_add(x, b) 
    return tf.nn.relu(x, name="myOutput")  

X = tf.Variable(tf.compat.v1.random_normal([784], name="myInput")) 
W = tf.Variable(tf.compat.v1.random_normal([5, 5, 1, 32])) 
B = tf.Variable(tf.compat.v1.random_normal([32])) 
x = tf.reshape(X, shape=[-1, 28, 28, 1]) 

export_dir='./model'  

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto()) as sess: 
    sess.run(tf.compat.v1.global_variables_initializer()) 
    y=conv2d(x,W,B) 
    sess.run([y]) 
    builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir) 
    signature = predict_signature_def(inputs={'myInput': X}, 
                                  outputs={'myOutput': y}) 
    builder.add_meta_graph_and_variables(sess=sess, 
                                     tags=["myTag"], 
                                     signature_def_map={'predict': signature}) 
    builder.save() 

2 Run the Python* script and check for the saved_model.pb stored at ./model location
python conv2D_fp32.py

3 Run the Auto Mixed Precision on Saved Model
3.1 Create gen_bf16_pb.py Python* Script
Initialize the graph_options with Auto Mixed Precision, load and convert the saved fp32 model.

from argparse import ArgumentParser 
from tensorflow.core.protobuf import config_pb2 
from tensorflow.core.protobuf import rewriter_config_pb2 
from tensorflow.python.grappler import tf_optimizer 
from tensorflow.python.tools import saved_model_utils 
import tensorflow as tf 
import time   

parser = ArgumentParser() 
parser.add_argument("input_dir", help="Input directory containing saved_model.pb.", type=str) 
parser.add_argument("as_text", help="Output graph in text protobuf format." 
                    "If False, would dump in binary format", type=bool) 
parser.add_argument("output_dir", help="Directory to store output graph.", type=str) 
parser.add_argument("output_graph", help="Output graph name. (e.g., foo.pb," 
                    "foo.pbtxt, etc)", type=str) 
args = parser.parse_args()   

graph_options = tf.compat.v1.GraphOptions(rewrite_options=rewriter_config_pb2.RewriterConfig(
     auto_mixed_precision_onednn_bfloat16=rewriter_config_pb2.RewriterConfig.ON)) 
optimizer_config = tf.compat.v1.ConfigProto(graph_options=graph_options) 
metagraph_def = saved_model_utils.get_meta_graph_def(args.input_dir, "myTag") 
output_graph = tf_optimizer.OptimizeGraph(optimizer_config, metagraph_def) 
tf.io.write_graph(output_graph, args.output_dir, args.output_graph, 
                  as_text=args.as_text) 

3.2 Run the Conversion Script
python gen_bf16_pb.py ./model True ./model saved_model_bf16.pbtxt

Console output:

2022-08-18 23:46:18.193892: I tensorflow/core/grappler/devices.cc:66] Number of eligible GPUs (core count >= 8, compute capability >= 0.0): 0
2022-08-18 23:46:18.194091: I tensorflow/core/grappler/clusters/single_machine.cc:358] Starting new session
2022-08-18 23:46:18.194400: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX512F AVX512_VNNI AVX512_BF16 AVX_VNNI AMX_TILE AMX_INT8 AMX_BF16 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2022-08-18 23:46:18.237297: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:2240] Converted 3/59 nodes to bfloat16 precision using 0 cast(s) to bfloat16 (excluding Const and Variable casts)

Again, notice 2 operators are converted into BFloat16 type. The script also supports dumping the model into protobuf binary file (.pb) or text file (.pbtxt).

Controlling the Operators that will be Ported to BFloat16 type Automatically
We provide some more details about Auto Mixed Precision for interested readers. An important point to note is that not all of TensorFlow’s* operators for CPU backend support BFloat16 type - this could be because either the support is missing (and is a WIP) or that the BFloat16 version of an operator may not offer much performance improvement over the FP32 version.

Furthermore, BFloat16 type for certain operators could lead to numerical instability of a neural network model. So we categorize TensorFlow* operators that are supported by MKL backend in BFloat16 type into 1) if they are always numerically stable, and 2) if they are always numerically unstable, and 3) if their stability could depend on the context. Auto Mixed Precision pass uses a specific Allow, Deny and Infer list of operators respectively to capture these operators. The exact lists could be found in auto_mixed_precision_lists.h file in TensorFlow* github repository.

We would like to mention that the default values of these lists already capture the most common BFloat16 usage models and to also ensure numerical stability of the model. There is, however, a way to add or remove operators from any of these lists as by setting environment variables that control these lists. For instance, executing

export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_ADD=Conv2D 

before running the model would add Conv2D operator to Deny list. While, executing

export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_REMOVE=Conv2D 

before running the model would remove Conv2D from Allow list. And executing both of these commands before running a model would move Conv2D from Allow list to Deny list.

In general, the template corresponding to the names of the environment variables controlling these lists is:

TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_${LIST}_${OP}=operator

where ${LIST} would be any of {ALLOW, DENY, INFER}, and ${OP} would be any of {ADD, REMOVE}.

To test this feature of adding an op into denylist and removing from the allowlist, run the code sample conv2D_bf16.py by enabling the environment variables with conv2d ops

export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_DENYLIST_ADD=Conv2D 
export TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_ALLOWLIST_REMOVE=Conv2D  
python conv2D_bf16.py 

As you can see from the console output and comparing with the Step 1, conv2D ops have been removed from the allowlist and added to the denylist, thus controlling the node from converting into BFloat16 equivalent.

2022-08-19 00:08:41.975709: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:2303] Running auto_mixed_precision_onednn_bfloat16 graph optimizer
2022-08-19 00:08:41.975944: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:1488] No allowlist ops found, nothing to do
2022-08-19 00:08:41.990983: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:2303] Running auto_mixed_precision_onednn_bfloat16 graph optimizer
2022-08-19 00:08:41.991196: I tensorflow/core/grappler/optimizers/auto_mixed_precision.cc:1488] No allowlist ops found, nothing to do

 


​
