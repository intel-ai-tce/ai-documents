​
Overview
This document provides details about new features and known issues for the Intel® AI Analytics Toolkit. The toolkit includes the following components:

Intel® Optimization for TensorFlow*

Intel® Extension for TensorFlow*

Intel® Optimization for PyTorch*

Intel® Extension for PyTorch*

Intel® oneCCL Bindings for PyTorch*

Intel® Distribution for Python*

Intel® Neural Compressor

Model Zoo for Intel® Architecture

Intel® Distribution of Modin*

Where to Find the Release
Please check the release page for more information on how to acquire the package.

Compatibility Notes
Component	Compatibility Version
Intel® Optimization for TensorFlow*	2.13
Intel® Extension for TensorFlow*	2.13
Intel® Optimization for PyTorch*	2.0.1
Intel® Extension for PyTorch*	2.0.110
Intel® oneCCL Bindings for PyTorch* 	2.0.100
Intel® Distribution for Python*	3.9
Intel® Distribution of Modin* 	0.23.0
Intel® Neural Compressor	2.2
Model Zoo for Intel® Architecture	2.12
What's New in AI Analytics Toolkit 2023.2
The NEW Intel AI Tools Selector beta to discover and install Intel's AI tools, libraries, and frameworks with greater flexibility to meet your needs, including recommendations to match common workflows and custom configurations.

Enhanced TensorFlow performance through a reduction in the number of “cast” operations for bfloat16 models using Intel® Advanced Matrix Extensions (AMX) with the Intel® Extension for TensorFlow.

Accelerate data preprocessing with Pandas 2.0 support in the Intel® Distribution for Modin, combining the latest advancements in Pandas with the benefits of parallel and distributed computing. 

Faster training and inference for AI workloads, increased hardware choice, simpler debug and diagnostics, and support for graph neural network processing, Intel® oneAPI Deep Neural Network Library (oneDNN) now has improved performance optimization for 4th Gen Intel® Xeon® Processors and Intel GPUs, extended diagnostics in verbose mode, and experimental support of Graph Compiler backend for Graph API. 

System Requirements
Please see system requirements.

How to Start Using the Tools
Please reference the usage guides for each of the included tools:

Intel® Optimization for TensorFlow* - See Performance Considerations
Intel® Optimization for PyTorch * - See Performance Considerations
Known Limitations
Runtime Out of Memory Error Using GPU
There is a potential known issue with the cleanup of resources at queue synchronization points in longer running jobs (most likely to show up in multi-tile or multi-device setups) that can lead to resources on the device being used up and causing out of memory errors.  

Workaround

In cases where this is identified, users can use the compiler hotfix available at below links to help address this situation. With this hotfix in place, cleanup of these resources will happen more often, i.e. after certain threshold of the number of allocated resources is hit. While a default value for this has been provided, users can enforce finer grain control through the usage of the environmental variable SYCL_PI_LEVEL_ZERO_COMMANDLISTS_CLEANUP_THRESHOLD.  The value is defined as: If non-negative, then the threshold is set to this value. If negative, the threshold is set to INT_MAX. Whenever the number of command lists in a queue exceeds this threshold, an attempt is made to cleanup completed command lists for their subsequent reuse. The default is 20

Compiler hotfix

Windows: https://registrationcenter-download.intel.com/akdlm/IRC_NAS/5c0da857-a4a2-4b7e-9914-4f2b0994c0ed/2023.1-windows-hotfx.zip

Linux: https://registrationcenter-download.intel.com/akdlm/IRC_NAS/89283df8-c667-47b0-b7e1-c4573e37bd3e/2023.1-linux-hotfix.zip

 
2023.1.1
Compatibility Notes

Component	Compatibility Version
Intel® Optimization for TensorFlow*	2.12
Intel® Extension for TensorFlow*	1.2
Intel® Optimization for PyTorch*	1.13
Intel® Extension for PyTorch*	1.13.120
Intel® oneCCL Bindings for PyTorch* 	1.13.120
Intel® Distribution for Python*	3.9
Intel® Distribution of Modin* 	0.19.0
Intel® Neural Compressor	2.1
Model Zoo for Intel® Architecture	2.11
What's New in AI Analytics Toolkit 2023.1.1

Intel® Neural Compressor optimizes auto-/multinode-tuning strategy & LLM memory.
Intel® Distribution of Modin introduces a new, experimental NumPy API.
Model Zoo for Intel® Architecture adds support for dataset downloader and data connectors.
Intel® Extension for TensorFlow now supports TensorFlow 2.12 and adds Ubuntu 22.04 and Red Hat Enterprise Linux 8.6 to the list of supported platforms.   
Intel® Extension for PyTorch is now compatible with oneDNN 3.1 which improves on PyTorch 1.13 operator coverage..
System Requirements

Please see system requirements.

How to Start Using the Tools

Please reference the usage guides for each of the included tools:

Intel® Optimization for TensorFlow* - See Performance Considerations
Intel® Optimization for PyTorch * - See Performance Considerations
Known Limitations

This release of Model Zoo does not support workloads on Intel® Data Center GPU Max Series, but supports workloads on Intel® Data Center GPU Flex Series.
Runtime Out of Memory Error Using GPU
There is a potential known issue with the cleanup of resources at queue synchronization points in longer running jobs (most likely to show up in multi-tile or multi-device setups) that can lead to resources on the device being used up and causing out of memory errors.  

Workaround

In cases where this is identified, users can use the compiler hotfix available at below links to help address this situation. With this hotfix in place, cleanup of these resources will happen more often, i.e. after certain threshold of the number of allocated resources is hit. While a default value for this has been provided, users can enforce finer grain control through the usage of the environmental variable SYCL_PI_LEVEL_ZERO_COMMANDLISTS_CLEANUP_THRESHOLD.  The value is defined as: If non-negative, then the threshold is set to this value. If negative, the threshold is set to INT_MAX. Whenever the number of command lists in a queue exceeds this threshold, an attempt is made to cleanup completed command lists for their subsequent reuse. The default is 20

Compiler hotfix

Windows: https://registrationcenter-download.intel.com/akdlm/IRC_NAS/5c0da857-a4a2-4b7e-9914-4f2b0994c0ed/2023.1-windows-hotfx.zip

Linux: https://registrationcenter-download.intel.com/akdlm/IRC_NAS/89283df8-c667-47b0-b7e1-c4573e37bd3e/2023.1-linux-hotfix.zip

 
2023.1
Compatibility Notes

Intel® Optimization for TensorFlow* is compatible with version 2.10.0

Intel® Extension for TensorFlow* is compatible with version 1.1.0

Intel® Optimization for PyTorch* is compatible with version 1.13.0

Intel® Extension for PyTorch* is compatible with version 1.13.10

Intel® oneCCL Bindings for PyTorch* is compatible with version 1.13.0 (cpu) or 1.13.100 (gpu)

Intel® Distribution for Python* is compatible with cpython version 3.9

Intel® Distribution of Modin* is compatible with version 0.17

Intel® Neural Compressor is compatible with version 1.14.2

Model Zoo for Intel® Architecture with version 2.8

What's New in AI Analytics Toolkit 2023.1

Accelerate your AI training and inference software performance using the 2023.1 AI Kit. In this release, Intel® Xe Matrix Extensions (Intel® XMX) in Intel Flex and Max Series GPUs are exposed to developers by Intel® oneAPI Deep Neural Network Library (oneDNN), through deep learning frameworks such as TensorFlow and PyTorch, for increased performance across wide range of market segments to deliver competitive performance.

Additional significant performance gains are provided by Intel® Extensions for TensorFlow (ITEX) and Intel® Extensions for PyTorch (IPEX), both having native GPU support.
System Requirements

Please see system requirements.

How to Start Using the Tools

Please reference the usage guides for each of the included tools:

Intel® Optimization for TensorFlow* - See Performance Considerations
Intel® Optimization for PyTorch * - See Performance Considerations
Known Limitation

This release of Model Zoo does not support workloads on Intel® Data Center GPU Max Series, but supports workloads on Intel® Data Center GPU Flex Series.
Intel® AI Analytics Toolkit (AI Kit) 2023.1 doesn't work with Intel® oneAPI Base Toolkit 2023.1 because of incompatibility with Intel® oneAPI DPC++/C++ Compiler and runtime and Intel® oneAPI Math Kernel Library. Please keep using Intel® oneAPI Base Toolkit 2023.0 with Intel AI Analytics Toolkit 2023.1.
 
2023.0
Compatibility Notes

Intel® Optimization for TensorFlow* is compatible with version 2.9.1

Intel® Optimization for PyTorch* is compatible with version 1.12.1

Intel® Distribution for Python* is compatible with cpython version 3.9

Intel® Distribution of Modin* is compatible with version 0.17

Intel® Neural Compressor is compatible with version 1.13

Model Zoo for Intel® Architecture with version 2.8

What's New in AI Analytics Toolkit 2023.0

Easily scale your AI solution from low compute resources to large or distributed compute resources with the new Heterogeneous data kernals (HDK) backend for Modin.
Use Accenture reference kits to jumpstart your AI solution.
Choose your platform, as AI Kit now supports running natively on Windows with full parity to Linux (except for distributed training).
Known Limitation

Bzip2 is required to extract packages, and a Linux system without this software will result in an installation failure.
Intel® Optimization for TensorFlow* compatible with version 2.9.1 may not include all the latest functional and security updates. A new version of Intel® Optimization for TensorFlow* is targeted to be released by March 2023 and will include additional functional and security updates. Customers should update to the latest version as it becomes available.
 
2022.3
Compatibility Notes

Intel® Optimization for TensorFlow* is compatible with version 2.9.1

Intel® Optimization for PyTorch* is compatible with version 1.12.1

Intel® Distribution for Python* is compatible with cpython version 3.9

Intel® Distribution of Modin* is compatible with version 0.13.3

Intel® Neural Compressor is compatible with version 1.13

Model Zoo for Intel® Architecture with version 2.8

What's New in AI Analytics Toolkit 2022.3

Product Review Windows OS support

Intel® Optimization for TensorFlow*

Updated to TensorFlow v2.9.1, and compiler options are no longer needed to enable the current oneDNN v2.6 optimizations on Intel Cascade lake and newer CPUs on Linux OS
Tensorflow now includes performance improvements for Bfloat16 models with AMX optimizations and more operations are supported with BF16 datatype
CVE and bug fixesl
Intel® Optimization for PyTorch*

Intel® Extension for PyTorch has been updated to 1.12.100, following PyTorch v1.12.
PyTorch now includes automatic int8 quantization and made it a stable feature. Runtime extension is stabilized and MultiStreamModule feature is brought to further boost throughput in offline inference scenario.
Enhancements in operation and graph has been added which are positive for performance of broad set of workloads.
Intel® Neural Compressor
Intel® Neural Compressor mainly updated with Tensorflow new quantization API support, QDQ quantization support for ITEX, mixed precision enhancement, DyNAS support, training for block-wise structure sparsity support, op-type wise tuning strategy support.
Intel® Neural Compressor improved productivity with lighter binary size, quantization accuracy diagnostic feature support by GUI, and experimental auto-coding support.
Intel® Model Zoo

Model Zoo for Intel® Architecture now supports Intel® Optimization for PyTorch  v1.12.1 and Intel® Optimization for TensorFlow v2.9.1, while also including fixes and improvements for product quality and use of current stable releases. Experimental support is added for SLES and SUSE platforms.
Intel® Distribution for Python*

 
2022.2
Compatibility Notes

Intel® Optimization for TensorFlow* is compatible with version 2.8

Intel® Optimization for PyTorch* is compatible with version 1.10

Intel® Distribution for Python* is compatible with cpython version 3.9

Intel® Distribution of Modin* is compatible with version 0.13

Intel® Neural Compressor is compatible with version 1.10.1

What's New

Product Review Windows OS support

Intel® Optimization for TensorFlow*

Performance improvements for Bfloat16 models with AMX optimizations.

Enabled support for 12th Gen Intel(R) Core (TM) (code named Alder Lake) platform.

No longer supports oneDNN block format, i.e., setting TF_ENABLE_MKL_NATIVE_FORMAT=0 will not enable blocked format.

To enable AMX optimization, you no longer need DNNL_MAX_CPU_ISA = AVX512_CORE_AMX.

Updated oneDNN to version 2.5.1

Improved _FusedMatMul operation, which enhances the performance of models like BERT

Added LayerNormalization ops fusion and BatchMatMul – Mul – AddV2 fusion to improve performance of Transformer based language models

Improved performance of EfficentNet and EfficientDet models with addition of swish (Sigmoid – Mul) fusion

 Removed unnecessary transpose elimination to enhance performance for 3DUnet model

Intel® Optimization for PyTorch*

changed the underhood device from XPU to CPU. With this change, the model and tensor do not need to be converted to the extension device to get a performance improvement.
optimize the Transformer* and CNN models by fusing more operators and applying NHWC.
Change the package name to intel_extension_for_pytorch while the original package name is intel_pytorch_extension.
support auto-mixed-precision for Bfloat16 data type.
provides the INT8 calibration as an experimental feature while it only supports post-training static quantization now.
Intel® Neural Compressor
Supported the quantization on latest deep learning frameworks
Supported the quantization for a new model domain (Audio)
Supported the compatible quantization recipes for framework upgrade
Supported fine-tuning and quantization using INC & Optimum for “Prune Once for All: Sparse Pre-Trained Language Models” published at ENLSP NeurIPS Workshop 2021
Proved the sparsity training recipes across multiple model domains (CV, NLP, and Recommendation System)
Improved INC GUI for easy quantization
Supported quantization on 300 random models
Added bare-metal examples for Bert-mini and DLRM
Intel® Model Zoo

Add support for TensorFlow v2.7.0 
Support for PyTorch v1.10.0 and IPEX v1.10.0
Intel® Distribution for Python*

Added new Diagnostics Utility for Intel® oneAPI Toolkits to diagnose the system status for using Intel® products. Learn more.
 
2022.1
Compatibility Notes

Intel® Optimization for TensorFlow* is compatible with version 2.6

Intel® Optimization for PyTorch* is compatible with version 1.8

Intel® Distribution for Python* is compatible with cpython version 3.9

Intel® Distribution of Modin* is compatible with version 0.8.2

Intel® Neural Compressor is compatible with version 1.7

What's New

Intel oneAPI Intel® AI Analytics Toolkit 2022.1.2 has been updated to include functional and security updates including Apache Log4j* version 2.17.1. Users should update to the latest version as it becomes available.

Intel® Optimization for PyTorch*

The Intel® Extension for PyTorch now supports Python 3.9 and Microsoft's Windows Subsystem for Linux (WSL).
Intel® Neural Compressor
Supported magnitude pruning on TensorFlow
Supported knowledge distillation on PyTorch

Supported multi-node pruning with distributed dataloader on PyTorch

Supported multi-node inference for benchmark on PyTorch

Improved quantization accuracy in SSD-Reset34 and MobileNet v3 on TensorFlow

Supported the configuration-free (pure Python) quantization
Improved ease-of-use user interface for quantization with few clicks
Added a domain-specific acceleration library for NLP models

Intel® Model Zoo

Add support for TensorFlow v2.6.0 and TensorFlow Serving v2.6.0
Add support for Pytorch 1.9 and Intel Extension for Pytorch 1.9
Add Transfer Learning sample IPython Notebook that fine-tunes BERT with the IMDb dataset
Add documentation to create an Intel Neural Compressor container with Intel Optimized TensorFlow
Additional models and precisions:
ML-Perf Transformer-LT Training (FP32 and BFloat16)
ML-Perf Transformer-LT Inference (FP32, BFloat16 and INT8)
ML-Perf 3D-Unet Inference (FP32, BFloat16 and INT8)
DIEN Training (FP32)
DIEN Inference (FP32 and BFloat16)
Intel® Distribution for Python* 

Intel® Distribution for Python now supports Python version 3.9

The dpctl package offers developers increased debugging capabilities with improved error handing and reporting

Data Parallel Python technology now provides zero copy data exchange performance across packages
Added new Diagnostics Utility for Intel® oneAPI Toolkits to diagnose the system status for using Intel® products. Learn more.
 Known Issue

Compatibility issue between Intel® AI Analytics Toolkit and Intel® Base Analytics Toolkit.
Intel Distribution for Python in Basekit 2022.2 Release updated three packages cryptography / pyopenssl / libprotobuf causing package confliction with TensorFlow in AI Kit 2022.1 Release.
Solution : Install AI Kit 2022.1 Release in a separate directory.
sudo ./l_AIKit_b_2021.1.8.618_offline.sh -s -a --install-dir /target/install/path --silent --eula accept



 
2021.4
Compatibility Notes

Intel® Optimization for TensorFlow* is compatible with version 2.5

Intel® Optimization for PyTorch * is compatible with version 1.8

Intel® Distribution for Python* is compatible with cpython version 3.7

Intel® Distribution of Modin* is compatible with version 0.8.2

Intel® Low Precision Optimization Tool  is compatible with version 1.5.1

What's New

Fine tune the performance of Natural Language algorithms through the latest sparsity and pruning features introduced in AI Analytics Toolkit.

Intel® Low Precision Optimization Tool
EGradient-sensitivity pruning for CNN model

Static quantization support for ONNX NLP model

Dynamic seq length support in NLP dataloader

Enrich quantization statistics

Intel® Model Zoo

Added links to Intel oneContainer Portal

Added documentation for running most workflows inside Intel® oneAPI AI Analytics Toolkit

Experimental support for running workflows onCentOS 8

Intel® Distribution for Python* 

Intel® Distribution for Python* release notes

System Requirements

Please see system requirements.

How to Start Using the Tools

Please reference the usage guides for each of the included tools:

Intel® Optimization for TensorFlow* - See Performance Considerations
Intel® Optimization for PyTorch * - See Performance Considerations
Known Limitation

Intel® Optimization for TensorFlow* 
Int8 will only work when environment variable TF_ENABLE_MKL_NATIVE_FORMAT=0 is set.
 
2021.3
Compatibility Notes

Intel® Optimization for TensorFlow* is compatible with version 2.5

Intel® Optimization for PyTorch * is compatible with version 1.8

Intel® Distribution for Python* is compatible with cpython version 3.7

Intel® Distribution of Modin* is compatible with version 0.8.2

Intel® Low Precision Optimization Tool  is compatible with version 1.4.1

What's New

Intel® Optimization for TensorFlow* 

​oneDNN - upgraded oneDNN to v2.2
The default for Intel TF is now native format, The user will need to set the env-variable TF_ENABLE_MKL_NATIVE_FORMAT=0 to use blocked formats.
OneDNN primitive cache enabled. Improved performance of models with batch size 1.
Various ops fusions with FP32, BFloat16, and INT8 data format
Conv2D+Squeeze+BiasAdd Fusion
MatMul+BiasAdd+Add Fusion.
Enabled MatMul + Bias + LeakyRelu Fusion.
CNMS performance optimization
Enabled DNNL CPU dispatch control.
Graph pattern match for grappler op fusion optimization
Supporting quantized pooling op for signed 8 bits.
Enable MklConv/MklFusedConv with explicit padding
Remove nGraph build support tensorflow#42870
Execute small gemm's single threaded.
Removed unnecessary OneDNN dependencies.
Removed DNNL 0.x support
Bug fixes
Issues resolved in TensorFlow 2.5
oneDNN resolved issues. 2.2 resolved issues
Fixed memory leak in MKLAddN
Fixed the bug to duplicate kernel registration of BatchMatMulV2.
Fixed unit test failures due to benchmark test API changes
incorrect result of _MKLMaxPoolGrad 40122.
Intel® Optimization for PyTorch * 
Upgraded the oneDNN from v1.5-rc to v1.8.1
Updated the README file to add the sections to introduce supported customized operators, supported fusion patterns, tutorials and joint blogs with stakeholders
Intel® Model Zoo
One new PyTorch workload containers and model packages that are available on the Intel® oneContainer Portal:

DLRM BFloat16 Training
Two new TF workload containers and model packages that are available on the Intel® oneContainer Portal:

3D U-Net FP32 Inference
SSD ResNet34 BFloat16 Training
Intel® Low Precision Optimization Tool
Extended Capabilities

Model conversion from QAT to Intel Optimized TensorFlow model
User Experience

More comprehensive logging message
UI enhancement with FP32 optimization, auto-mixed precision (BF16/FP32), and graph visualization
Online document: https://intel.github.io/lpot
Model Zoo

INT8 key models updated (BERT on TensorFlow, DLRM on PyTorch, etc.)
20+ HuggingFace model quantization
Pruning

Pruning/sparsity API refinement
Magnitude-based pruning on PyTorch
Quantization
PyTorch FX-based quantization support
TensorFlow & ONNX RT quantization enhancement
Intel® Distribution for Python* 
Intel® Distribution for Python* release notes
Known Limitation

Intel® Optimization for TensorFlow* 
Int8 will only work when environment variable TF_ENABLE_MKL_NATIVE_FORMAT=0 is set.
 
2021.2
Compatibility Notes

Intel® Optimization for TensorFlow* is compatible with version 2.3

Intel® Optimization for PyTorch * is compatible with version 1.7

Intel® Distribution for Python* is compatible with cpython version 3.7

Intel® Distribution of Modin* is compatible with version 0.8.2

What's New

Intel® Optimization for TensorFlow* 

​oneDNN - moved from 0.x to version 1.4

Support for Intel® MKL-DNN version 0.x is still available.
Issues resolved in TensorFlow 2.3
oneDNN resolved issues
1.4 resolved issues
1.3 resolved issues
1.2 resolved issues
Intel® Optimization for PyTorch * 

​New PyTorch * 1.7.0 was newly supported by Intel extension for PyTorch *.

Device name was changed from DPCPP to XPU.

Enabled the launcher for end users.

Improvement for INT8 optimization with refined auto mixed precision API.

More operators are optimized for the int8 inference and bfp16 training of some key workloads, like: MaskRCNN, SSD-ResNet34, DLRM, RNNT.

New custom operators: ROIAlign, RNN, FrozenBatchNorm, nms.

Performance improvement for several operators: tanh, log_softmax, upsample, embeddingbad and enables int8 linear fusion.

Bug fixes

Intel® Model Zoo

Several new TensorFlow* and PyTorch* models added to the Intel® Model Zoo.
Ten new TensorFlow workload containers and model packages that are available on the Intel® oneContainer Portal
Two new PyTorch workload containers and model packages that are available on the Intel® oneContainer Portal
Three new TensorFlow Kubernetes packages that are available on the Intel® oneContainer Portal:
A new Helm chart to deploy TensorFlow Serving on a K8s cluster
Bug-fixes, improvements to documentations
Intel® Low Precision Optimization Tool

New backends (PyTorch/IPEX, ONNX Runtime) preview support
Add built-in industry dataset/metric and custom registration
Preliminary input/output node auto-detection on TensorFlow models
New INT8 quantization recipes: bias correction and label balance
30+ OOB models validated
Known Limitation

2021.2 AI Kit installation causes a Intel® oneapi Base Toolkit 2021.3 installation issue if users don't have 2021.2 Base Toolkit installation alongside the 2021.2 AI Kit installation.
To avoid this issue, Users could install Intel® oneapi Base Toolkit 2021.2 alongside the 2021.2 AI Kit installation first, and then Intel® oneapi Base Toolkit 2021.3 could be installed successfully.
Intel® Optimization for PyTorch* 
Multi-node training still encounter hang issues after several iterations. The fix will be included in the next official release.
 
2021.1
Compatibility Notes

Intel® Optimization for TensorFlow* is compatible with version 2.2

Intel® Optimization for PyTorch * is compatible with version 1.5

Intel® Distribution for Python* is compatible with cpython version 3.7

Intel® Distribution of Modin* is compatible with version 0.8.2

Known Limitation

Intel® Optimization for PyTorch* requires Intel® AVX512 supports on the system.

Notices and Disclaimers

Intel technologies may require enabled hardware, software or service activation.

No product or component can be absolutely secure.

Your costs and results may vary.

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.

No license (express or implied, by estoppel or otherwise) to any intellectual property rights is granted by this document.

The products described may contain design defects or errors known as errata which may cause the product to deviate from published specifications. Current characterized errata are available on request.

Intel disclaims all express and implied warranties, including without limitation, the implied warranties of merchantability, fitness for a particular purpose, and non-infringement, as well as any warranty arising from course of performance, course of dealing, or usage in trade.

​
