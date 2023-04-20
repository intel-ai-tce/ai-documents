​
This document provides a summary of new and changed product features.

Where to Find the Release
Please follow the steps to download Intel® oneAPI Base toolkit contained oneDNN from the Main Portal of Intel® oneAPI Base toolkit, and follow the installation instructions to install.

2023.1
What's New
Performance Optimizations
Intel® Architecture processors:
Improved performance for 4th generation Intel® Xeon Scalable processor (formerly Sapphire Rapids).
Introduced initial optimizations for future Intel® Xeon Scalable processor (code name Sierra Forest). The functionality is disabled by default and should be enabled via CPU dispatcher control..
Intel® Processor Graphics and Xe architecture-based Graphics::
Improved performance for Intel® Data Center GPU Max Series (formerly Ponte Vecchio).
Improved performance for Intel® Arc graphics (formerly Alchemist and DG2) and Intel® Data Center GPU Flex Series (formerly Arctic Sound-M).
Improved concat primitive performance with per-argument scales on Intel® GPUs.
Functionality
Enabled Graph API as a production feature. Graph API is intended to simplify oneDNN integration into frameworks.
Added an option to zero-out weight gradient in RNN primitive. See details in corresponding RFC.
Added support for the non-zero alpha parameter in the batch normalization ReLU post-op on Intel® GPUs.
Enabled the layer normalization primitive with f64 datatype support on Intel® GPUs.
Deprecated Functionality
Legacy CPU-only configurations are deprecated and will be removed in oneDNN 2024 release.
Previous Releases
 
2023.0
What's New

Deliver production quality AI Deep Learning optimizations for 4th Gen Intel® Xeon® Scalable processor, Intel® Xeon® processor Max Series, Intel® Data Center GPU Flex Series, and Intel® Arc™ A-Series GPUs
With support for S8/S8 weights and activations enable greater input influence on the outcomes on 4th Gen Intel® Xeon® Scalable processor with Intel® Advanced Matrix Extensions (Intel® AMX) acceleration instruction set
Support wider operators -BF32 on 4th Gen Intel® Xeon® Scalable processor and TF32 Intel® Data Center GPU Flex Series and , Intel® Max Series GPUs for more accurate inferencing
Enable limited support for FP64 operators on Intel® Data Center GPU Max Series GPUs for high precision model deployment
Deliver experimental Graph API support (opensource only) to simplify integration to frameworks and extend optimization capabilities
Performance Optimizations

Intel® Architecture processors:
Improved performance for 4th generation Intel Xeon Scalable processor (formerly Sapphire Rapids).
Introduced performance optimizations for bf16 floating point math mode on 4th generation Intel Xeon Scalable processors (code name Sapphire Rapids). The bf16 math mode allows oneDNN to use bf16 arithmetic and Intel AMX instructions in computations on fp32 data.
Introduced FP16 support and initial optimizations for future Intel Xeon Scalable processor (code name Granite Rapids).
Intel® Processor Graphics and Xe architecture-based Graphics::
Improved performance for Intel Data Center GPU Max Series (formerly Ponte Vecchio).
Introduced performance optimizations for tf32 floating point math mode on future Xe Architecture graphics (code name Ponte Vecchio). The tf32 math mode allows oneDNN to use tf32 arithmetic in computations on fp32 data.
Improved performance for Intel Arc graphics (formerly Alchemist and DG2) and Intel Data Center GPU Flex Series (formerly Arctic Sound-M).
Functionality

Introduced runtime output scales support in all primitives.
Introduced scales support in concat primitive.
Extended floating point math mode API with tf32 data type option.
Extended eltwise primitive with support for hardsigmoid algorithm.
Extended layer normalization primitive with support for mixed source and destination data types.
Extended depthwise post-op with support for arbitrary padding size. The implementation is available only on Intel processors.
Added limited fp64 data type support in convolution primitive. Optimized implementation is available for future Xe Architecture graphics (code name Ponte Vecchio).
Extended int8 convolution and deconvolution implementations on GPUs with arbitrary destination data type support.
Extended batch normalization primitive with dnnl_fuse_norm_add_relu flag that allows to fuse sum and relu operations. The implementation is available for Intel GPUs.
Extended GPU deconvolution primitive implementation with support for output scales and zero points.
Introduced new quantization scheme. Major changes include support for per-argument runtime scales in all primitives and unquantized bias.
Introduced support for Intel DPC++/C++ Compiler 2023.0, including new features from the SYCL 2020 standard.
Extended persistent cache to cover GPU engine object. This improvement allows applications to further reduce oneDNN initialization time.
Extended threadpool API with a function to indicate maximum available concurrency.
Extended binary primitive implementation on GPU with bfloat16 source and int8 destination support.
Usability

Added matmul_perf example that benchmarks matmul primitive for all supported data types.
Introduced annotations for JIT kernels to allow profilers like Linux perf to correctly label JIT code.
Extended verbose logs converter with RNN primitive support.
Added verbose output for dnnl_*gemm* calls.
Removed Level Zero headers from the list of build time dependencies.
Extended the set of supported format tags to cover formats used in applications.
Deprecated Functionality

Support for SYCL 1.2.1 (aka SYCL 2017 standard) is deprecated and will be removed in the future releases.
Static output scales are deprecated and will be removed in the next release.
Convolution Winograd algorithm implementation for int8 data type is deprecated and will be removed in the next release.
Breaking Changes

Changed formula for AUGRU RNN cell to align with Tensorflow. See proposal for details.
Removed deprecated APIs.
Removed operation descriptor object and made memory descriptor object opaque. See details in operation and memory descriptors RFC.
Removed creation time primitive scales support and primitive output scales support. See details in quantization scaling RFC.
Removed support for Intel DPC++/C++ Compiler with SYCL 1.2.1 (aka SYCL 2017) standard.
Removed Winograd convolution implementation for int8 data type.

 
2022.2
No change from 2022.1 version to 2022.2 version.


 
2022.1
Performance Optimizations

Intel® Processor Graphics and Xe architecture-based Graphics:
Improved performance for future Xe Architecture graphics (code name Ponte Vecchio).
Improved performance for future Arc graphics (code name Alchemist and DG2).
Intel® Architecture processors
Improved performance for future Intel Xeon Scalable processors (code name Sapphire Rapids). The functionality is now enabled by default and requires Linux kernel 5.16 or later.
Improved performance of matmul primitive for processors with Intel AVX-512 support.
New Functionality

Introduced bfloat16 destination support for int8 convolution, matmul and inner product primitives for processors with Intel AVX-512 support and or future Intel Xeon® Scalable processors (code name Sapphire Rapids)
Extended RNN primitive with support for AUGRU cell.
Added support for non-zero negative slope in ReLU post-op for batch normalization primitive.
Introduced support for mixed source and destination data types in softmax primitive.
Introduced persistent cache API. This functionality allows to serialize and reuse JIT kernels.
Usability

Reduced stack consumption in GEMM implementation.
Breaking Changes

Removed performance optimizations for Intel Xeon Phi processors. oneDNN will continue to be functional on these processors using Intel AVX2 codepath..
Deprecated Functionality

Support for SYCL 1.2.1 (aka SYCL 2017 standard) is deprecated and will be removed in future releases.
Known issues and limitations

See DPC++ limitations that impact the library as well.
 
2022.0
Performance Optimizations

Intel® Processor Graphics and Xe architecture-based Graphics:
Introduced initial optimizations for future Xe Architecture graphics (code name Ponte Vecchio).
Improved pooling and layer normalization primitives performance.
Intel® Architecture processors
Improved performance for future Intel Xeon Scalable processors (code name Sapphire Rapids). The functionality is now enabled by default and requires Linux kernel 5.16.
Improved performance of matmul primitive for processors with Intel® Advanced Vector Extensions 512 (Intel® AVX-512) support.
New Functionality

Introduced support for compiler with SYCL 2020 standard support.
Introduced support for the ICX/ICPX and DPCPP compiler drivers available in the Intel® oneAPI DPC++ Compiler.
Usability

Added environment variables and build options with 'ONEDNN' prefix.
Breaking Changes

The Intel MKL-DNN compatibility API is removed. See Transition from Intel® MKL-DNN to oneDNN page for instructions on moving to the new API.
Deprecated Functionality

Support for Intel® Xeon Phi processors is deprecated and will be removed in the next release.
Support for SYCL 1.2.1 (aka SYCL 2017 standard) is deprecated and will be removed in future releases.
Known issues and limitations

See DPC++ limitations that impact the library as well.

 
2021.4
Performance Optimizations

Improved primitive cache performance for Intel Graphics products.
Intel® Processor Graphics and Xe architecture-based Graphics:
Introduced initial optimizations for future Intel® Arc™ Graphics codenamed Alchemist (ACM). That includes optimizations of compute-bound primitives (Convolution, GEMM) for s8/u8, f16 and bf16 datatypes via DPAS (Dot Product Systolic Accumulate) instructions.
Improved performance of convolution and deconvolution after some OpenCL kernels were re-implemented using kernel code generator (jit:ir implementation as reported by DNNL_VERBOSE).
Intel® Architecture processors
Improved performance for future Intell® Xeon Scalable processor (code name Sapphire Rapids). The functionality is disabled by default and should be enabled via CPU dispatcher control.
Improved binary primitive performance for cases when one of the tensors is broadcasted.
Improved reorder primitive performance for memory formats with padding and/or zero points.
Improved performance of reduction primitive, reorder, shuffle primitives. 
Improved performance of depthwise forward convolution primitive for processors with Intel® AVX512 support.
Improved performance of forward inner product primitive for the shapes with minibatch equal to 1 for processors with Intel® AVX512 support.
Improved int8 GEMM performance for processors with Intell® AVX2 and Intel® DL Boost support.
New Functionality

Introduced PReLU post-op support in convolution and matmul.
Extended maximum allowed post-ops chain for compute primitives (convolution, deconvolution, inner product, and matmul) to 32.
Introduced support for zero points in sum post-op for convolution and matmul. The functionality is implemented only for CPUs.
Extended binary primitive with support for mixed data types for input tensors. The functionality is implemented only for CPUs.
Extended sum post-op for convolution and matmul primitives with support for mixed data types. The functionality is implemented only for CPUs.
Usability

Reduced overall library size by trimming down use of templates, OpenCL headers, and TBB headers. The configurations that benefitted the most are CPU only configuration with TBB threading.
Deprecated Functionality

Intel MKL-DNN compatibility API is deprecated and will be removed in the next update. See Transition from Intel MKL-DNN to oneDNN page for instructions on moving to new API.
Support for Intel Xeon Phi processors is deprecated and will be removed in the next release.
Known issues and limitations

See DPC++ limitations that impact the library as well.

 
2021.3
Performance Optimizations

Extended primitive cache to improve primitive descriptor creation performance.
Improved primitive cache performance in multithreaded configurations.
Intel® Processor Graphics and Xe architecture-based Graphics:
Introduced initial optimizations for bfloat16 compute functionality for future Intel Xeon Scalable processor (code name Sapphire Rapids). The functionality is disabled by default and should be enabled via CPU dispatcher control.
Improved performance of binary primitive and binary post-op for cases with broadcast and mixed source and destination formats.
Improved performance of reduction primitive.
Improved performance of depthwise convolution primitive with NHWC activations for training cases
Intel® Architecture processors
Introduced initial optimizations for bfloat16 functionality for future Intel® Xeon Scalable processor with Intel® AMX support (code name Sapphire Rapids). The functionality is disabled by default and should be enabled via CPU dispatcher control.
Improved performance of int8 compute functionality for future Intel® Xeon Scalable processor (code name Sapphire Rapids). The functionality is disabled by default and should be enabled via CPU dispatcher control. 
Introduced initial performance optimizations for future Intel® Core processor with Intel® AVX2 and Intel® DL Boost instructions support (code name Alder Lake).
Improved performance of int8 primitives for processors with Intel® SSE4.1 instruction set support.
Improved performance of int8 and bfloat16 RNN and inner product primitives.
Introduced CPU ISA hints environment variable and API. New API is intended to dispatch function implementations using YMM registers to improve performance on processors with a single Intel® AVX512 compute unit.
Improved forward convolution performance for Intel® AVX-512 systems.
Improved convolution and batch normalization performance with threadpool.
Improved performance of bfloat16 shuffle primitive.
Improved performance of `dnnl_gemm` and functionality relying on this implementation for cases with `n=1` on all supported processors.

New Functionality

Extended batch normalization and layer normalization primitives API to take separate scale and shift arguments.
Extended resampling primitive with post-ops support and mixed source and destination data types..
Usability

Introduced support for DPC++ debug configuration on Windows
Breaking changes

Updated minimal supported CMake version from to 2.8.12 (was 2.8.11)
Known issues and limitations

Backward inner product primitive may produce incorrect result for the shapes with number of output channels not been multiple by 16 for future Intel Xeon Scalable processor (code name Sapphire Rapids)
Convolution with binary post-op may produce incorrect results for formats with channel padding.
Pooling and batch normalization primitives may hang on Windows GEN9 and DG1 in DPC++/L0 configuration.
Pooling and batch normalization primitives with 4D double blocked memory formats may produce NaNs or hang on Linux DG1 platforms.
See DPC++ limitations that impact the library as well.

 
2021.2
Performance Optimizations

Reduced overheads associated with primitive cache.
Intel® Processor Graphics and Xe architecture-based Graphics:
Improved performance of int8 primitives with NHWC activations format.
Improved functionality performance for padded memory formats.
Improved performance of reorder and shuffle primitives for multiple formats and all dimensions.
Improved performance of fp16 pooling primitive.
Improved performance of lnorm primitive for plain memory formats.
Improved performance of resampling primitive for blocked memory formats .
Improved performance of Winograd convolution.
Intel® Architecture processors
Introduced initial optimizations for bfloat16 functionality for future Intel® Xeon Scalable processor with Intel® AMX support (code name Sapphire Rapids). The functionality is disabled by default and should be enabled via CPU dispatcher control.
Improved performance of int8 compute functionality for future Intel® Xeon Scalable processor (code name Sapphire Rapids). The functionality is disabled by default and should be enabled via CPU dispatcher control. 
Introduced initial performance optimizations for future Intel® Core processor with Intel® AVX2 and Intel® DL Boost instructions support (code name Alder Lake).
Improved performance of int8 primitives for processors with Intel® SSE4.1 instruction set support.
Improved performance of int8 and bfloat16 RNN and inner product primitives.
Introduced CPU ISA hints environment variable and API. New API is intended to dispatch function implementations using YMM registers to improve performance on processors with a single Intel® AVX512 compute unit.
Improved forward convolution performance for Intel® AVX-512 systems.
Improved convolution and batch normalization performance with threadpool.
Improved performance of bfloat16 shuffle primitive.
Improved performance of `dnnl_gemm` and functionality relying on this implementation for cases with `n=1` on all supported processors.

New Functionality

Introduced binary post-op for (de)-convolution, pooling, eltwise, binary, inner product, matmul and reduction (GPU only) along with performance optimizations for CPUs and GPUs. Extended the number of supported post-ops for primitives to 20.
Extended eltwise primitive with support for `logsigmoid`, `mish`, `hardswish`, and `clip_v2` algorithms.
Introduced support for PRelu primitive. 
Introduced int8 support for LSTM primitive with projection for CPU.
Introduced asymmetric quantization support for int8 deconvolution.
Extended matmul implementation with support for per-output channel zero-points for quantization.
Extended support for broadcasting in binary primitive to both inputs for CPU.
Extended binary primitive with support for comparison operators.
Introduced float16 support in reduction primitive for GPU.
Introduced support for mixed input and output types in binary primitive for GPU.
Introduced support for post-ops in GPU resampling implementation.
Usability

Added API to enable displaying timestamps in oneDNN verbose mode. Timestamps allow to use oneDNN verbose output in profiling tools.
Improved presentation of oneDNN primitives in  Intel® VTune™ Profiler.
Validation

Extended benchdnn to report operation bandwidth.
Added ability to choose target GPU in benchdnn.

Known issues and limitations

When using driver version older than 27.20.100.9316 for Intel® UHD Graphics for 9th Gen Intel® Processor on Windows, convolution/de-convolution functions may sporadically hang or produce incorrect results in DPC++ configuration with LevelZero. Please upgrade your driver version to fix the issue. An alternative solution is to use DPC++ with OpenCL backend with DPC++ compiler.
Reorder, prelu, softmax, and pooling primitives on GPUs may be slower for zero padded memory formats than Intel oneDNN 2021.1.
Reorder operation for 5D tensor with two dimensions equal to 16 and one uneven dimension can produce incorrect results on Intel® Iris® Xe Max Graphics.
Eltwise primitive may produce incorrect results for oneDNN DPC++ configuration with Level Zero runtime. In order to avoid this, use DPC++ with OpenCL backend with DPC++ compiler.
Deconvolution primitive may segfault with int8 data on processors for cases with non-trivial padding on processors with Intel AVX-512 support.
Deconvolution primitive may segault with int8 data when used with post-ops and per_oc broadcast on processors with Intel AVX2 support.
Pooling, batch normalization, and binary primitives may segfault when executed on Xe architecture-based graphics. No workaround available.
Non-Intel GPUs are not supported. The library API allows to create a DNNL engine by index (the order of devices is determined by the SYCL runtime), and there is no check for GPU devices being non-Intel. To have more control, users can create a DNNL engine passing SYCL device and context explicitly.
When running GPU kernels that take longer than a certain time (it depends on OS and system settings), you may face a situation resulting in apparent hang of the application. There are ways to configure driver or system settings to disable this timeout and avoid hanging of DPC++ or OpenCL programs, including oneDNN examples:
On Linux* (See more details at OpenCL™ Driver for Intel® HD, Iris™, and Iris™ Pro Graphics for Linux):
$ sudo bash -c 'echo N > /sys/module/i915/parameters/enable_hangcheck'

On Windows* (See more details at Timeout Detection and Recovery (TDR) Registry Keys):
Increase TdrDelay and TdrDdiDelay values in registry

See DPC++ limitations that impact the library as well.
 
2021.1
New Functionality

Introduced SYCL API extensions compliant with oneAPI specification v1.0.
Introduced support for Intel® oneAPI DPC++/C++ compiler.
Introduced Unified Shared Memory (USM) support for Intel Processor Graphics and Xe architecture-based graphics.
Known issues and limitations

Pooling, batch normalization, and binary primitives may segfault when executed on Xe architecture-based graphics. No workaround available.
Non-Intel GPUs are not supported. The library API allows to create a DNNL engine by index (the order of devices is determined by the SYCL runtime), and there is no check for GPU devices being non-Intel. To have more control, users can create a DNNL engine passing SYCL device and context explicitly.
When running GPU kernels that take longer than a certain time (it depends on OS and system settings), you may face a situation resulting in apparent hang of the application. There are ways to configure driver or system settings to disable this timeout and avoid hanging of DPC++ or OpenCL programs, including oneDNN examples:
On Linux* (See more details at OpenCL™ Driver for Intel® HD, Iris™, and Iris™ Pro Graphics for Linux):
$ sudo bash -c 'echo N > /sys/module/i915/parameters/enable_hangcheck'

On Windows* (See more details at Timeout Detection and Recovery (TDR) Registry Keys):
Increase TdrDelay and TdrDdiDelay values in registry

See DPC++ limitations that impact the library as well.


Notices and Disclaimers

Intel technologies may require enabled hardware, software or service activation.

No product or component can be absolutely secure.

Your costs and results may vary.

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.

No license (express or implied, by estoppel or otherwise) to any intellectual property rights is granted by this document.

The products described may contain design defects or errors known as errata which may cause the product to deviate from published specifications. Current characterized errata are available on request.

Intel disclaims all express and implied warranties, including without limitation, the implied warranties of merchantability, fitness for a particular purpose, and non-infringement, as well as any warranty arising from course of performance, course of dealing, or usage in trade.

​
