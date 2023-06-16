# Overview
The Intel® oneAPI Collective Communications Library (oneCCL) enables developers and researchers to more quickly train newer and deeper models. This is done by using optimized communication patterns to distribute model training across multiple nodes.

The library is designed for easy integration into deep learning (DL) frameworks, whether you are implementing them from scratch or customizing existing ones.

- Built on top of lower-level communication middleware - MPI and OFI (libfabrics) which transparently support many interconnects, such as Cornelis Networks*, InfiniBand*, and Ethernet.
- Optimized for high performance on Intel® CPUs and GPUs.
- Allows the tradeoff of compute for communication performance to drive scalability of communication patterns.
- Enables efficient implementations of collectives that are heavily used for neural network training, including allreduce, and allgather.
# 2021.9 Release
## Major Features Supported
Table1

| Functionality	| Subitems|	CPU	| GPU |
| ------------- | --------| --- | --- |
| Collective operations |	Allgatherv |	X	| X |
||	Allreduce |	X |	X |
|| 	Alltoall	| X	| X |
|| 	Alltoallv	| X	| X |
|| 	Barrier	| X	| X |
|| 	Broadcast	| X | X |
|| 	Reduce	| X	| X |
|| 	ReduceScatter	| X | X |
|Data types |	[u]int[8, 16, 32, 64] |	X	| X |
|| 	fp[16, 32, 64], bf16 |X |	X |
|Scaling |	Scale-up |	X |	X |
| |	Scale-out |	X |	X |
| Programming model	| Rank = device	| 1 rank per process |	1 rank per process |

## Service functionality
- Interoperability with SYCL*:
  - Construction of oneCCL communicator object based on SYCL context and SYCL device
  - Construction of oneCCL stream object based on SYCL queue
  - Construction of oneCCL event object based on SYCL event
  - Retrieving of SYCL event from oneCCL event associated with oneCCL collective operation
  - Passing SYCL buffer as source/destination parameter of oneCCL collective operation
## What's New
- Improved scaling efficiency of the Scaleup algorithms for Alltoall and Allgather
- Add collective selection for scaleout algorithm for device (GPU) buffers
## System Requirements
please see [system requirements](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/oneapi-collective-communication-library-system-requirements.html).
 
# Previous Releases

<details>
<summary>2021.8</summary>
<br>

## Major Features Supported

Table1
| Functionality	| Subitems|	CPU	| GPU |
| ------------- | --------| --- | --- |
| Collective operations |	Allgatherv |	X	| X |
||	Allreduce |	X |	X |
|| 	Alltoall	| X	| X |
|| 	Alltoallv	| X	| X |
|| 	Barrier	| X	| X |
|| 	Broadcast	| X | X |
|| 	Reduce	| X	| X |
|| 	ReduceScatter	| X | X |
|Data types |	[u]int[8, 16, 32, 64] |	X	| X |
|| 	fp[16, 32, 64], bf16 |X |	X |
|Scaling |	Scale-up |	X |	X |
| |	Scale-out |	X |	X |
| Programming model	| Rank = device	| 1 rank per process |	1 rank per process |

## Service functionality

- Interoperability with SYCL*:
  - Construction of oneCCL communicator object based on SYCL context and SYCL device
  - Construction of oneCCL stream object based on SYCL queue
  - Construction of oneCCL event object based on SYCL event
  - Retrieving of SYCL event from oneCCL event associated with oneCCL collective operation
  - Passing SYCL buffer as source/destination parameter of oneCCL collective operation

## What's New
- Provides optimized performance for Intel® Data Center GPU Max Series utilizing oneCCL.
- Enables support for Allreduce, Allgather, Reduce, and Alltoall connectivity for GPUs on the same node
## Known issues and limitations
- The 'using namespace oneapi;' directive is not recommended, as it may result in compilation errors 
when oneCCL is used with other oneAPI libraries. You can instead create a namespace alias for oneCCL, e.g. 
 
namespace oneccl = ::oneapi::ccl;
oneccl::allreduce(...);

- Limitations imposed by Intel® oneAPI DPC++ Compiler:
  - SYCL buffers cannot be used from different queues  

</details> 

<details>
<summary>2021.7</summary>
<br>
 
# 2021.7.1 Release
Intel® oneAPI Collective Communications Library 2021.7.1 has been updated to include functional and security updates. Users should update to the latest version as it becomes available.

# 2021.7 Release

## Major Features Supported

Table1
| Functionality	| Subitems|	CPU	| GPU |
| ------------- | --------| --- | --- |
| Collective operations |	Allgatherv |	X	| X |
||	Allreduce |	X |	X |
|| 	Alltoall	| X	| X |
|| 	Alltoallv	| X	| X |
|| 	Barrier	| X	| X |
|| 	Broadcast	| X | X |
|| 	Reduce	| X	| X |
|| 	ReduceScatter	| X | X |
|Data types |	[u]int[8, 16, 32, 64] |	X	| X |
|| 	fp[16, 32, 64], bf16 |X |	X |
|Scaling |	Scale-up |	X |	X |
| |	Scale-out |	X |	X |
| Programming model	| Rank = device	| 1 rank per process |	1 rank per process |

## Service functionality

- Interoperability with SYCL*:
  - Construction of oneCCL communicator object based on SYCL context and SYCL device
  - Construction of oneCCL stream object based on SYCL queue
  - Construction of oneCCL event object based on SYCL event
  - Retrieving of SYCL event from oneCCL event associated with oneCCL collective operation
  - Passing SYCL buffer as source/destination parameter of oneCCL collective operation
## What's New
- no change from previous release.

## Known issues and limitations

- Limitations imposed by Intel® oneAPI DPC++ Compiler:
  - SYCL buffers cannot be used from different queues
- The 'using namespace oneapi;' directive is not recommended, as it may result in compilation errors 
when oneCCL is used with other oneAPI libraries. You can instead create a namespace alias for oneCCL, e.g. 
```c++  
namespace oneccl = ::oneapi::ccl;
oneccl::allreduce(...);
```

</details>

<details>
<summary>2021.6</summary>
<br>

  
## Major Features Supported

Table1
| Functionality	| Subitems|	CPU	| GPU |
| ------------- | --------| --- | --- |
| Collective operations |	Allgatherv |	X	| X |
||	Allreduce |	X |	X |
|| 	Alltoall	| X	| X |
|| 	Alltoallv	| X	| X |
|| 	Barrier	| X	| X |
|| 	Broadcast	| X | X |
|| 	Reduce	| X	| X |
|| 	ReduceScatter	| X | X |
|Data types |	[u]int[8, 16, 32, 64] |	X	| X |
|| 	fp[16, 32, 64], bf16 |X |	X |
|Scaling |	Scale-up |	X |	X |
| |	Scale-out |	X |	X |
| Programming model	| Rank = device	| 1 rank per process |	1 rank per process |

## Service functionality

- Interoperability with SYCL*:
  - Construction of oneCCL communicator object based on SYCL context and SYCL device
  - Construction of oneCCL stream object based on SYCL queue
  - Construction of oneCCL event object based on SYCL event
  - Retrieving of SYCL event from oneCCL event associated with oneCCL collective operation
  - Passing SYCL buffer as source/destination parameter of oneCCL collective operation

## What's New

- Intel® oneAPI Collective Communications Library now supports Intel® Instrumentation and Tracing Technology (ITT) profiling
- Intel® oneAPI Collective Communications Library can be seamlessly integrated with Windows platforms with WSL2 (Windows Subsystem for Linux 2) support
- Enhanced application stability with runtime dependency check for Level Zero, in Intel® oneAPI Collective Communications Library
## Known issues and limitations
- Limitations imposed by Intel® oneAPI DPC++ Compiler:
  - SYCL buffers cannot be used from different queues
- The 'using namespace oneapi;' directive is not recommended, as it may result in compilation errors 
when oneCCL is used with other oneAPI libraries. You can instead create a namespace alias for oneCCL, e.g. 
```c++  
namespace oneccl = ::oneapi::ccl;
oneccl::allreduce(...);
```

</details>

<details>
<summary>2021.5</summary>
<br>
 
## What's New
- Added support for output SYCL event to track status of CCL operation
- Added OFI/verbs provider with dmabuf support into package
- Bug fixes
## Known issues and limitations

- Limitations imposed by Intel®  DPC++ compiler:
  - SYCL buffers cannot be used from different queues
- The 'using namespace oneapi;' directive is not recommended, as it may result in compilation errors 
when oneCCL is used with other oneAPI libraries. You can instead create a namespace alias for oneCCL, e.g. 
```c++  
namespace oneccl = ::oneapi::ccl;
oneccl::allreduce(...);
```
</details>

<details>
<summary>2021.4</summary>
<br>
  
## What's New
- Memory binding of worker threads is now supported
- NIC filtering by name is now supported for OFI-based multi-NIC
- IPv6 is now supported for key-value store (KVS)
## Known issues and limitations
- Limitations imposed by Intel®  DPC++ compiler:
  - SYCL buffers cannot be used from different queues
- The 'using namespace oneapi;' directive is not recommended, as it may result in compilation errors 
when oneCCL is used with other oneAPI libraries. You can instead create a namespace alias for oneCCL, e.g. 
```c++  
namespace oneccl = ::oneapi::ccl;
oneccl::allreduce(...);
```
 
</details>

<details>
<summary>2021.3</summary>
<br>
  
## What's New

- Added OFI-based multi-NIC support
- Added OFI/psm3 provider support
- Bug fixes
## Known issues and limitations

- Limitations imposed by Intel®  DPC++ compiler:
  - SYCL buffers cannot be used from different queues
- The 'using namespace oneapi;' directive is not recommended, as it may result in compilation errors 
when oneCCL is used with other oneAPI libraries. You can instead create a namespace alias for oneCCL, e.g. 
```c++  
namespace oneccl = ::oneapi::ccl;
oneccl::allreduce(...);
```
 
</details>

<details>
<summary>2021.2</summary>
<br>
  
## What's New
- Added float16 datatype support.
- Added ip-port hint for customization of KVS creation.
- Optimized communicator creation phase.
- Optimized multi-GPU collectives for single-node case.
- Bug fixes

## Known issues and limitations

- Limitations imposed by Intel®  DPC++ compiler:
  - SYCL buffers cannot be used from different queues
- The 'using namespace oneapi;' directive is not recommended, as it may result in compilation errors 
when oneCCL is used with other oneAPI libraries. You can instead create a namespace alias for oneCCL, e.g. 
```c++  
namespace oneccl = ::oneapi::ccl;
oneccl::allreduce(...);
```
 
</details>

<details>
<summary>2021.1</summary>
<br>
  
## What's New
- Added [u]int16 support
- Added initial support for external launch mechanism
- Fixed bugs

## Known issues and limitations

- Limitations imposed by Intel®  DPC++ compiler:
  - SYCL buffers cannot be used from different queues
- The 'using namespace oneapi;' directive is not recommended, as it may result in compilation errors 
when oneCCL is used with other oneAPI libraries. You can instead create a namespace alias for oneCCL, e.g. 
```c++  
namespace oneccl = ::oneapi::ccl;
oneccl::allreduce(...);
```
 
</details>

**Notices and Disclaimers**

Intel technologies may require enabled hardware, software or service activation.

No product or component can be absolutely secure.

Your costs and results may vary.

© Intel Corporation. Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.

No license (express or implied, by estoppel or otherwise) to any intellectual property rights is granted by this document.

The products described may contain design defects or errors known as errata which may cause the product to deviate from published specifications. Current characterized errata are available on request.

Intel disclaims all express and implied warranties, including without limitation, the implied warranties of merchantability, fitness for a particular purpose, and non-infringement, as well as any warranty arising from course of performance, course of dealing, or usage in trade.

​
