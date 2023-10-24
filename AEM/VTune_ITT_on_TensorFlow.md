PROFILING TENSORFLOW\* WORKLOADS WITH THE INSTRUMENTATION AND TRACING TECHNOLOGY (ITT) AP
=======================================================

In this recipe, you will learn:

* What is Intel® VTune™ Profiler

* What is Instrumentation and Tracing Technology (ITT) API

* How ITT feature benefit TensorFlow Workloads profiling

* Simple sample showcasing how to profile with ITT feature with the Intel oneAPI AI ToolKit Docker Image




What is Intel® VTune™ Profiler
--------------------
Intel® VTune™ Profiler is a performance analysis tool for serial and multithreaded applications. For those who are familiar with Intel Architecture, Intel® VTune™ Profiler provides a rich set of metrics to help users understand how the application executed on Intel platforms, and thus have an idea where the performance bottleneck is.  


What is Instrumentation and Tracing Technology (ITT) API
--------------------
For deep learning programmers, function hotspots might not help a lot for analyzing their deep learning workloads. Primitives level or deep learning operations level hotspots mean more to deep learning developers. 
In order to provide primitives level or DL operations level information to DL developers, we introduce instrumentation and tracing technology APIs to generate and control the collection of the VTune trace data of oneDNN primitive execution, and the feature supports both CPU and GPU.  

The Instrumentation and Tracing Technology API (ITT API) provided by the Intel® VTune™ Profiler enables target application to generate and control the collection of trace data during its execution.  
The advantage of ITT feature is to label time span of individual TensorFlow operators, as well as customized regions, on Intel® VTune™ Profiler GUI. When users find anything abnormal, it will be very helpful to locate which operator behaved unexpectedly.  

How ITT feature benefit TensorFlow Workloads profiling
--------------------

### 1. get the primitives timeline chart from VTune, and identify any protential performance issue.  
For below diagrams, you can see convolution, inner production, eltwise and reorder primitives are tagged among 
threads in the timeline chart. You could identify that there are two reorder operations between eltwise and convolution,  
those reorder ops could be further reduced to improve performance.
<img width="1000" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/6fb1844e-53fa-4e6e-aadc-f40840730839">


### 2. get platform information such as L1/L2 cache miss or level of FP vectorization on primitive level   
For below diagram, users can group profiling results by Task Types, and then VTune will group information by
oneDNN primitives tagged as different Task Types.   
Therefore, users can see platform level detail information like FP vectorization on primitives level.  
Users can understand performance issues among each oneDNN primitives.  
![group](https://github.com/intel-ai-tce/ai-documents/assets/21761437/478d8e68-e372-4f3e-baa1-045bb5aeeed4)

### 3. map primitive with related computation kernels.  
By tagging primitives, users can also map primitive with computation kernels.  
For example, users can understand one inner primitive contains several GPU gen9_gemm_nocopy_f32 kernels.  
![ops_mapping](https://github.com/intel-ai-tce/ai-documents/assets/21761437/caaecea4-ac65-44da-bbaf-659b76c3701e)


Simple sample showcasing how to profile with ITT feature
--------------------
We use a [simple TensorFlow workload](https://github.com/oneapi-src/oneAPI-samples/blob/master/AI-and-Analytics/Getting-Started-Samples/IntelTensorFlow_GettingStarted/TensorFlow_HelloWorld.py) with only convolution and relu operations from oneAPI sample github, and also use oneAPI AI Kit docker image with VTune and TensorFlow conda environment installed.  


### Environment Setup
In this example, we use oneapi-aikit 2023.2 docker image, so users need to use below command to pull the docker image.
After those commands, users will be in the bash shell with all AI Kit components ready for disposal. 
```
wget https://raw.githubusercontent.com/oneapi-src/oneAPI-samples/master/AI-and-Analytics/Getting-Started-Samples/IntelAIKitContainer_GettingStarted/run_oneapi_docker.sh
chmod +x run_oneapi_docker.sh
./run_oneapi_docker.sh intel/oneapi-aikit:2023.2.0-devel-ubuntu22.04
```
In 2023.2 AI Kit, we have TensorFlow 2.13, and we need to upgrade to TensorFlow 2.14 to have this ITT new feature.
USers could follow below commands to upgrade TensorFlow version in the docker instance.

```
conda create --name tensorflow-2.14 --clone tensorflow
source activate tensorflow-2.14
pip uninstall intel-tensorflow
pip install tensorflow
```


### Profile simple TensorFlow sample with different profiling type
There are different [VTune analysis types](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-2/running-command-line-analysis.html) and we demostrate how to use ITT feature among different profiling types below with the TensorFlow simple workload.

First, get the simple TensorFlow workload and activate tensorflow-2.14 conda environment
```
wget https://raw.githubusercontent.com/oneapi-src/oneAPI-samples/master/AI-and-Analytics/Getting-Started-Samples/IntelTensorFlow_GettingStarted/TensorFlow_HelloWorld.py
source activate tensorflow-2.14
```

#### 1. HotSpot Analysis
Analyze application flow and identify sections of code that take a long time to execute (hotspots).  
Use below command to profile the TensorFlow workload with HotSpot profiling type.  
```
vtune -collect hotspots -data-limit=5000 -knob sampling-mode=hw -knob sampling-interval=0.1  -result-dir r001hs -quiet  python TensorFlow_HelloWorld
```
Once users finish profiling, users could use below command to launch the VTune Web UI and use web browser to view the result.
```
vtune-backend --allow-remote-access option --data-directory=./
```

We suggest to customize grouping for Bottom-up Tab, and group by "Task Domain / Task Type / Function / Call Stack".
Users could refer to [menu customize grouping](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-2/menu-customize-grouping.html) for detail instructions.  
For HotSpot profiling type, users could mainly focus on Task Time and Task Count.
<img width="1000" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/72912f3b-4d8f-4b0e-9a39-ae3bad1cface">


#### 2. Threading Analysis
Collect data on how an application is using available logical CPU cores, discover where parallelism is incurring synchronization overhead, identify where an application is waiting on synchronization objects or I/O operations, and discover how waits affect application performance.  
Use below command to profile the TensorFlow workload with Threading profiling type. 
```
ulimit -n 128000; vtune -collect threading -data-limit=5000 -knob sampling-and-waits=hw  -knob sampling-interval=0.1 -result-dir r001tr -quiet  python TensorFlow_HelloWorld.py
```
Once users finish profiling, users could use below command to launch the VTune Web UI and use web browser to view the result.
```
vtune-backend --allow-remote-access option --data-directory=./
```
We suggest to customize grouping for Bottom-up Tab, and group by "Task Domain / Task Type / Function / Call Stack".
Users could refer to [menu customize grouping](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-2/menu-customize-grouping.html) for detail instructions.  
For Threading profiling type, users could mainly focus on Spin Time, Preemption Wait Time and Sunc Wait Time.
<img width="1000" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/1650d20c-9a87-4d9c-b467-6ad04a7bc6f9">


#### 3. Microarchitecture Exploration Analysis
Collect hardware events for analyzing a typical client application. This analysis calculates a set of predefined ratios used for the metrics and facilitates identifying hardware-level performance problems.  
Use below command to profile the TensorFlow workload with Microarchitecture Exploration profiling type. 
```
 vtune -collect uarch-exploration -data-limit=5000 -knob sampling-interval=0.1  -result-dir r001ue -quiet  python TensorFlow_HelloWorld.py
```
Once users finish profiling, users could use below command to launch the VTune Web UI and use web browser to view the result.
```
vtune-backend --allow-remote-access option --data-directory=./
```
We suggest to customize grouping for Bottom-up Tab, and group by "Task Domain / Task Type / Function / Call Stack".
Users could refer to [menu customize grouping](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-2/menu-customize-grouping.html) for detail instructions.  
For Microarchitecture Exploration profiling type, users could mainly focus on L1, L2, L3, DRAM Bound and Core Bound..
<img width="1000" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/6a245d50-f425-4991-b828-975bbe87855f">


#### 4. High Performance Compute Analysis
Identify opportunities to optimize CPU, memory, and FPU utilization for compute-intensive or throughput applications. 
Use below command to profile the TensorFlow workload with Microarchitecture Exploration profiling type.  
```
vtune -collect hpc-performance -data-limit=5000  -knob sampling-interval=0.1   -result-dir r001hpe -quiet  python TensorFlow_HelloWorld.py
```
Once users finish profiling, users could use below command to launch the VTune Web UI and use web browser to view the result.
```
vtune-backend --allow-remote-access option --data-directory=./
```
We suggest to customize grouping for Bottom-up Tab, and group by "Task Domain / Task Type / Function / Call Stack".
Users could refer to [menu customize grouping](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-2/menu-customize-grouping.html) for detail instructions.  
For High Performance Compute profiling type, users could mainly focus on Spin Time, Memory Bound, and NUMA Remote Accesses.
<img width="1000" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/72faaa18-a699-4912-92a8-3725f70b33db">

#### 5. Memory Access Analysis
Identify memory-related issues, like NUMA problems and bandwidth-limited accesses, and attribute performance events to memory objects (data structures), which is provided due to instrumentation of memory allocations/de-allocations and getting static/global variables from symbol information.  
Use below command to profile the TensorFlow workload with Memory Acces profiling type.  
```
vtune -collect memory-access  -knob sampling-interval=0.1  -result-dir r001ma -quiet  python TensorFlow_HelloWorld.py
```
Once users finish profiling, users could use below command to launch the VTune Web UI and use web browser to view the result.  
```
vtune-backend --allow-remote-access option --data-directory=./
```
We suggest to customize grouping for Bottom-up Tab, and group by "Task Domain / Task Type / Function / Call Stack".  
Users could refer to [menu customize grouping](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-2/menu-customize-grouping.html) for detail instructions.  
For Memory Access profiling type, users could mainly focus on  Memory Bound, and LLC Miss Count.   
<img width="1000" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/46fe75a5-27f2-4238-801d-39805800e74c">


#### 6. Memory Consumption Analysis
Analyze memory consumption by your Linux application, its distinct memory objects and their allocation stacks.  
Use below command to profile the TensorFlow workload with Memory Consumption profiling type.  
```
vtune -collect memory-consumption  -data-limit=5000     -result-dir r001mc -quiet  python TensorFlow_HelloWorld.py
```
Once users finish profiling, users could use below command to launch the VTune Web UI and use web browser to view the result.  
```
vtune-backend --allow-remote-access option --data-directory=./
```
We suggest to customize grouping for Bottom-up Tab, and group by "Task Domain / Task Type / Function / Call Stack".  
Users could refer to [menu customize grouping](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-2/menu-customize-grouping.html) for detail instructions.  
For Memory Consumption profiling type, users could mainly focus on Allocation/Deallocation Delta.   
<img width="700" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/5052fc66-939c-427a-b83e-1f51c8a60fc4">



