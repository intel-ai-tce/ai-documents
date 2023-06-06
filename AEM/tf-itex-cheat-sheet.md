Get started with Intel® Optimization for TensorFlow* and Intel® Extension for TensorFlow* using the following commands.

# Intel® Optimization for TensorFlow*: A Public Release from Google®
Features and optimizations for TensorFlow* on Intel hardware are frequently upstreamed and included in stock TensorFlow* releases. As of TensorFlow* v2.9, Intel® oneAPI Deep Neural Network Library (oneDNN) optimization is automatically enabled.

For more information, see TensorFlow.

Basic Installation Using PyPI*

pip install tensorflow

Basic Installation Using Anaconda*

conda install -c conda-forge tensorflow

Import TensorFlow

import tensorflow as tf

Capture a Verbose Log (Command Prompt)

export ONEDNN_VERBOSE=1

Parallelize Execution (in the Code)

tf.config.threading.set_intra_op_parallelism_threads()

tf.config.threading.set_intra_op_parallelism_threads()

tf.config.set_soft_device_placement(enabled=1)

Parallelize Execution (Command Prompt)

export TF_NUM_INTEROP_THREADS=<number of physical cores per socket>

export TF_NUM_INTRAOP_THREADS=<number of sockets>

Non-Uniform Memory Access (NUMA)

numactl --cpunodebind N --membind N python <script>

Enable bf16 Training

from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_bfloat16')

mixed_precision.set_global_policy(policy)



Intel® Optimization for TensorFlow*: A Public Release from Intel®
In addition to the performance tuning options listed under the Google® public release, the Intel® public release offers OpenMP* optimizations for further performance enhancements.

For additional installation methods, see the Intel® Optimization for TensorFlow* Installation Guide.

For more information about performance, see Maximize TensorFlow* Performance on CPU.

Basic Installation Using PyPI*

pip install intel-tensorflow

Basic Installation Using Anaconda*

conda install tensorflow (Linux/MacOS)

conda install tensorflow-mkl (Windows)

Import TensorFlow

Import tensorflow as tf

Capture a Verbose Log (Command Prompt)

export ONEDNN_VERBOSE=1

Parallelize Execution (in the Code)

tf.config.threading.set_intra_op_parallelism_threads()

tf.config.threading.set_intra_op_parallelism_threads()

tf.config.set_soft_device_placement(enabled=1)

Parallelize Execution (Command Prompt)

export TF_NUM_INTEROP_THREADS=<number of physical cores per socket>

export TF_NUM_INTRAOP_THREADS=<number of sockets>

Non-Uniform Memory Access (NUMA)

numactl --cpunodebind N --membind N python <script>

Enable bf16 Training

from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_bfloat16')

mixed_precision.set_global_policy(policy)

Set the Maximum Number of Threads (Command Prompt)

export OMP_NUM_THREADS=num physical cores

Bind OpenMP Threads to Physical Processing Units

export KMP_AFFINITY=granularity=fine,compact,1,0

Set a Wait Time (ms) After Completing the Execution of a Parallel Region Before Sleeping

export KMP_BLOCKTIME=<time>

Recommended to be to 0 for CNN or 1 for non-CNN (user should verify empirically)

Print an OpenMP Runtime Library Env Variables During Execution

export KMP_SETTINGS=TRUE



Intel® Extension for TensorFlow*
This extension provides the most up-to-date features and optimizations on Intel® hardware, most of which will eventually be upstreamed to stock TensorFlow* releases. Additionally, while users can get many optimization benefits by default without needing an additional set up, Intel® Extension for TensorFlow* provides further tuning and custom operations to boost performance even more.

For additional installation methods, see the Intel® Extension for TensorFlow* Installation Guide.

For more information, see Intel® Extension for TensorFlow*.

Basic GPU Installation using PyPI

pip install --upgrade intel-extension-for-tensorflow[gpu]

Import Intel Extension for TensorFlow

import intel_extension_for_tensorflow as itex

Get an XPU Back End Type

itex.get_backend()

Toggle a GPU Back End (in the Code): Set by Default

itex.set_backend(‘GPU’)

Toggle a GPU Back End (Command Prompt): Set by Default

Export ITEX_XPU_BACKEND="GPU"

Advanced Automatic Mixed Precision (in the Code): A Basic Configuration with Improved Inference Speed with Reduced Memory Consumption

auto_mixed_precision_options = itex.AutoMixedPrecisionOptions()



auto_mixed_precision_options.data_type = itex.BFLOAT16 # or itex.FLOAT16

Advanced Automatic Mixed Precision (Command Prompt): A Basic Configuration with Improved Inference Speed with Reduced Memory Consumption

export ITEX_AUTO_MIXED_PRECISION=1 

export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="BFLOAT16" # or "FLOAT16"

Customized AdamW Optimizer (in the Code)

itex.ops.AdamWithWeightDecayOptimizer(

    weight_decay_rate=0.001,

    learning_rate=0.001, beta_1=0.9,

    beta_2=0.999,

    epsilon=1e-07, name='Adam',

    exclude_from_weight_decay=["LayerNorm",

    "layer_norm", "bias"], **kwargs

)



Customized Layer Normalization (in the Code)

itex.ops.LayerNormalization(

    axis=-1, epsilon=0.001, center=True,

    scale=True,

    beta_initializer='zeros',

    gamma_initializer='ones',

    beta_regularizer=None,

    gamma_regularizer=None,

    beta_constraint=None,

    gamma_constraint=None, **kwargs

)

Customized GELU (in the Code)

itex.ops.gelu(

    features, approximate=False, name=None

)

Customized LSTM (in the Code)

itex.ops.ItexLSTM(

    200, activation='tanh',

    recurrent_activation='sigmoid',

    use_bias=True,

    kernel_initializer='glorot_uniform',

    recurrent_initializer='orthogonal',

    bias_initializer='zeros', **kwargs

)



For more information and support, or to report any issues, see:

Intel® Extension for TensorFlow* Issues on GitHub*

TensorFlow* Issues on GitHub

Intel® AI Analytics Toolkit Forum



Sign up and try this extension for free using Intel® Developer Cloud for oneAPI.
