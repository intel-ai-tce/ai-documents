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

### Environment Setup

```
wget https://raw.githubusercontent.com/oneapi-src/oneAPI-samples/master/AI-and-Analytics/Getting-Started-Samples/IntelAIKitContainer_GettingStarted/run_oneapi_docker.sh
chmod +x run_oneapi_docker.sh
./run_oneapi_docker.sh intel/oneapi-aikit:2023.2.0-devel-ubuntu22.04
```




### Profile simple TensorFlow sample with different profiling type
[VTune analysis types](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-2/running-command-line-analysis.html)

```
vtune-backend --allow-remote-access option --data-directory=./
```
#### HotSpot Analysis
```
vtune -collect hotspots -data-limit=5000 -knob sampling-mode=hw -knob sampling-interval=0.1  -result-dir r001hs -quiet  python TensorFlow_HelloWorld
```

<img width="1000" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/72912f3b-4d8f-4b0e-9a39-ae3bad1cface">


#### Threading Analysis
```
ulimit -n 128000; vtune -collect threading -data-limit=5000 -knob sampling-and-waits=hw  -knob sampling-interval=0.1 -result-dir r005tr -quiet  python TensorFlow_HelloWorld.py
```
<img width="1000" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/1650d20c-9a87-4d9c-b467-6ad04a7bc6f9">


#### Microarchitecture Exploration Analysis
```
 vtune -collect uarch-exploration -data-limit=5000 -knob sampling-interval=0.1  -result-dir r008ue -quiet  python TensorFlow_HelloWorld.py
```
<img width="1000" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/6a245d50-f425-4991-b828-975bbe87855f">


#### High Performance Compute Analysis
```
vtune -collect hpc-performance -data-limit=5000  -knob sampling-interval=0.1   -result-dir r010hpe -quiet  python TensorFlow_HelloWorld.py
```
<img width="1000" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/72faaa18-a699-4912-92a8-3725f70b33db">

#### Memory Access Compute Analysis
```
vtune -collect memory-access  -knob sampling-interval=0.1  -result-dir r011ma -quiet  python TensorFlow_HelloWorld.py
```
<img width="1000" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/46fe75a5-27f2-4238-801d-39805800e74c">


#### Memory Consumption Compute Analysis
```
vtune -collect memory-consumption  -data-limit=5000     -result-dir r009mc -quiet  python TensorFlow_HelloWorld.py
```
<img width="700" alt="image" src="https://github.com/intel-ai-tce/ai-documents/assets/21761437/5052fc66-939c-427a-b83e-1f51c8a60fc4">



#### Get Intel® Optimization for TensorFlow\* Pre-Built Images

<details>
  <summary>Install the latest Intel® Optimization for TensorFlow\* from Anaconda\* Cloud</summary>
  <br>
Available for Linux\*, Windows\*, MacOS\*

| **OS** | **TensorFlow\* version** | 
| -------- | -------- | 
| Linux\* | 2.12.0 | 
| Windows\*| 2.10.0 | 
| MacOS\* | 2.12.0 | 



</details>

