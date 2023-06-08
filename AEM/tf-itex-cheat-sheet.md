Get started with Intel® Optimization for TensorFlow* and Intel® Extension for TensorFlow* using the following commands.
<br><br>
## Intel® Optimization for TensorFlow*: A Public Release from Google
Features and optimizations for TensorFlow* on Intel hardware are frequently upstreamed and included in stock TensorFlow* releases. As of TensorFlow* v2.9, Intel® oneAPI Deep Neural Network Library (oneDNN) optimization is automatically enabled.

For more information, see [TensorFlow](https://www.tensorflow.org/).

<table><tbody>
<tr><td>Basic Installation Using PyPI*</td><td>pip install tensorflow</td></tr>
<tr><td>Basic Installation Using Anaconda*</td><td>conda install -c conda-forge tensorflow</td></tr>
<tr><td>Import TensorFlow</td><td>import tensorflow as tf</td></tr>
<tr><td>Capture a Verbose Log (Command Prompt)</td><td>export ONEDNN_VERBOSE=1</td></tr>
<tr><td>Parallelize Execution (in the Code)</td><td>tf.config.threading.set_intra_op_parallelism_threads(&lt;number of physical cores per socket&gt;) <br> tf.config.threading.set_inter_op_parallelism_threads(&lt;number of sockets&gt;) <br> tf.config.set_soft_device_placement(True) <br># Users could tune the INTRAOP and INTEROP setting based on the workloads</td></tr>
<tr><td>Parallelize Execution (Command Prompt)</td><td>export TF_NUM_INTRAOP_THREADS=&lt;number of physical cores per socket&gt; <br> export TF_NUM_INTEROP_THREADS=&lt;number of sockets&gt; <br># Users could tune the INTRAOP and INTEROP setting based on the workloads</td></tr>
<tr><td>Non-Uniform Memory Access (NUMA)</td><td>numactl --cpunodebind N --membind N python &lt;script&gt;</td></tr>
<tr><td>Enable Keras Mixed Precision with BF16</td><td>from tf.keras import mixed_precision <br> mixed_precision.set_global_policy('mixed_bfloat16')</td></tr>
</tbody></table>

<br><br>
## Intel® Optimization for TensorFlow*: A Public Release from Intel
In addition to the performance tuning options listed under the Google public release, the Intel public release offers OpenMP* optimizations for further performance enhancements.

For additional installation methods, see the [Intel® Optimization for TensorFlow* Installation Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html).

For more information about performance, see the [Maximize TensorFlow* Performance on CPU](https://www.intel.com/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html).

<table><tbody>
<tr><td>Basic Installation Using PyPI*</td><td>pip install intel-tensorflow</td></tr>
<tr><td>Basic Installation Using Anaconda*</td><td>conda install tensorflow (Linux/MacOS) <br> conda install tensorflow-mkl (Windows)</td></tr>
<tr><td>Import TensorFlow</td><td>import tensorflow as tf</td></tr>
<tr><td>Capture a Verbose Log (Command Prompt)</td><td>export ONEDNN_VERBOSE=1</td></tr>
<tr><td>Parallelize Execution (in the Code)</td><td>tf.config.threading.set_intra_op_parallelism_threads(&lt;number of physical cores per socket&gt;) <br> tf.config.threading.set_inter_op_parallelism_threads(&lt;number of sockets&gt;) <br> tf.config.set_soft_device_placement(True) <br># Users could tune the INTRAOP and INTEROP setting based on the workloads</td></tr>
<tr><td>Parallelize Execution (Command Prompt)</td><td>export TF_NUM_INTRAOP_THREADS=&lt;number of physical cores per socket&gt; <br> export TF_NUM_INTEROP_THREADS=&lt;number of sockets&gt; <br># Users could tune the INTRAOP and INTEROP setting based on the workloads</td></tr>
<tr><td>Non-Uniform Memory Access (NUMA)</td><td>numactl --cpunodebind N --membind N python &lt;script&gt;</td></tr>
<tr><td>Enable Keras Mixed Precision with BF16</td><td>from tf.keras import mixed_precision <br> mixed_precision.set_global_policy('mixed_bfloat16')</td></tr>
<tr><td>Set the Maximum Number of Threads (Command Prompt)</td><td>export OMP_NUM_THREADS=&lt;number of physical cores per socket&gt;</td></tr>
<tr><td>Bind OpenMP Threads to Physical Processing Units</td><td>export KMP_AFFINITY=granularity=fine,compact,1,0</td></tr>
<tr><td>Set a Wait Time (ms) After Completing the Execution of a Parallel Region Before Sleeping</td><td>export KMP_BLOCKTIME=&lt;time&gt; <br> # Recommended to be to 0 for CNN or 1 for non-CNN (user should verify empirically)</td></tr>
<tr><td>Print an OpenMP Runtime Library Env Variables During Execution</td><td>export KMP_SETTINGS=TRUE</td></tr>
</tbody></table>

<br><br>
## Intel® Extension for TensorFlow*
This extension provides the most up-to-date features and optimizations on Intel hardware, supporting both Intel CPU and Intel GPU devices, most of which will eventually be upstreamed to stock TensorFlow* releases. Additionally, while users can get many optimization benefits by default without needing an additional set up, Intel® Extension for TensorFlow* provides further tuning and custom operations to boost performance even more.

For additional installation methods, see the [Intel® Extension for TensorFlow* Installation Guide](https://intel.github.io/intel-extension-for-tensorflow/latest/docs/install/installation_guide.html).

For more information, see [Intel® Extension for TensorFlow*](https://intel.github.io/intel-extension-for-tensorflow/latest/get_started.html).

<table><tbody>
<tr><td>Basic Installation Using PyPI*</td><td>pip install --upgrade intel-extension-for-tensorflow[gpu] # Install for GPU <br><br> pip install --upgrade intel-extension-for-tensorflow[cpu] # Install for CPU [Experimental]</td></tr>
<tr><td>Import Intel® Extension for TensorFlow*</td><td>import intel_extension_for_tensorflow as itex</td></tr>
<tr><td>Get the Current XPU Backend Type</td><td>itex.get_backend()</td></tr>
<tr><td>Set the Specific Backend Type (in the Code): Set by Default</td><td>itex.set_backend('GPU') # 'CPU' </td></tr>
<tr><td>Set the Specific Backend Type (Command Prompt): Set by Default</td><td>export ITEX_XPU_BACKEND="GPU" # "CPU"</td></tr>
<tr><td>Advanced Automatic Mixed Precision (in the Code): A Basic Configuration with Improved Inference Speed with Reduced Memory Consumption</td><td>auto_mixed_precision_options = itex.AutoMixedPrecisionOptions() <br> auto_mixed_precision_options.data_type = itex.BFLOAT16 #itex.FLOAT16 <br><br> graph_options = itex.GraphOptions(auto_mixed_precision_options=auto_mixed_precision_options) <br> graph_options.auto_mixed_precision = itex.ON <br><br> config = itex.ConfigProto(graph_options=graph_options) <br> itex.set_config(config)</td></tr>
<tr><td>Advanced Automatic Mixed Precision (Command Prompt): A Basic Configuration with Improved Inference Speed with Reduced Memory Consumption</td><td>export ITEX_AUTO_MIXED_PRECISION=1 <br> export ITEX_AUTO_MIXED_PRECISION_DATA_TYPE="BFLOAT16" # or "FLOAT16"</td></tr>
<tr><td>Customized AdamW Optimizer (in the Code)</td><td>itex.ops.AdamWithWeightDecayOptimizer( <br>&emsp;&emsp;weight_decay_rate=0.001, learning_rate=0.001, beta_1=0.9, beta_2=0.999, <br>&emsp;&emsp;epsilon=1e-07, name='Adam', <br>&emsp;&emsp;exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"], **kwargs <br>)</td></tr>
<tr><td>Customized Layer Normalization (in the Code)</td><td>itex.ops.LayerNormalization( <br>&emsp;&emsp;axis=-1, epsilon=0.001, center=True, scale=True, <br>&emsp;&emsp;beta_initializer='zeros', gamma_initializer='ones', <br>&emsp;&emsp;beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, <br>&emsp;&emsp;gamma_constraint=None, **kwargs <br>)</td></tr>
<tr><td>Customized GELU (in the Code)</td><td>itex.ops.gelu( <br>&emsp;&emsp;features, approximate=False, name=None <br>)</td></tr>
<tr><td>Customized LSTM (in the Code)</td><td>itex.ops.ItexLSTM( <br>&emsp;&emsp;200, activation='tanh', <br>&emsp;&emsp;recurrent_activation='sigmoid', <br>&emsp;&emsp;use_bias=True, <br>&emsp;&emsp;kernel_initializer='glorot_uniform', <br>&emsp;&emsp;recurrent_initializer='orthogonal', <br>&emsp;&emsp;bias_initializer='zeros', **kwargs <br>)</td></tr>
</tbody></table>

<br>
For more information and support, or to report any issues, see:

<br>[Intel® Extension for TensorFlow* Issues on GitHub*](https://github.com/intel/intel-extension-for-tensorflow/issues)

[TensorFlow* Issues on GitHub*](https://github.com/tensorflow/tensorflow/issues)

[Intel® AI Analytics Toolkit Forum](https://community.intel.com/t5/Intel-oneAPI-AI-Analytics/bd-p/ai-analytics-toolkit)

<br>Sign up and try this extension for free using [Intel® Developer Cloud for oneAPI](https://devcloud.intel.com/oneapi/get_started).
