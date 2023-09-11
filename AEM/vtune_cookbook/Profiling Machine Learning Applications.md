**Profiling Machine Learning Applications (NEW)**

*Learn how to use Intel® VTune™ Profiler to profile Machine Learning (ML) workloads.*

In our increasingly digital world powered by software and web-based applications, Machine Learning (ML) applications have become extremely popular. The ML community uses several Deep Learning (DL) frameworks like Tensorflow\*, PyTorch\*, and Keras\* to solve real world problems.

However, understanding computational and memory bottlenecks in DL code like Python or C++ is challenging and often requires significant effort due to the presence of hierarchical layers and non-linear functions. Frameworks like Tensorflow\* and PyTorch\* provide native tools and APIs that enable the collection and analysis of performance metrics during different stages of Deep Learning Model development. But the scope of these profiling APIs and tools is quite limited. They do not provide deep insight at the hardware level to help you optimize different operators and functions in the Deep Learning models.

In this recipe, learn how you can use VTune Profiler to profile a Python workload and improve data collection with additional APIs.

**Content Expert: [Rupak Roy](https://community.intel.com/t5/user/viewprofilepage/user-id/183427)**

- [INGREDIENTS](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-2/profiling-machine-learning-applications.html#INGR)
- DIRECTIONS:
  - [Run VTune Profiler on a Python Application](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-2/profiling-machine-learning-applications.html#DIRECT)
  - [Include Intel® Instrumentation and Tracing Technology (ITT)-Python APIs](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-2/profiling-machine-learning-applications.html#ITT)
  - [Run Hotspots and Microarchitecture Exploration Analyses](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-2/profiling-machine-learning-applications.html#HOTSPOTS)
  - [Add PyTorch* ITT APIs (for PyTorch Framework only)](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-2/profiling-machine-learning-applications.html#PYTORCH)
  - [Run Hotspots Analysis with PyTorch ITT APIs](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-2/profiling-machine-learning-applications.html#HSA-PYTORCH)

Ingredients

Here are the hardware and software tools you need for this recipe.

- **Application**: This recipe uses the [TensorFlow_HelloWorld.py](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/IntelTensorFlow_GettingStarted) and [Intel_Extension_For_PyTorch_Hello_World.py](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics/Getting-Started-Samples/Intel_Extension_For_PyTorch_GettingStarted) applications. Both of these code samples are implementations of a simple neural network with a convolution layer, a normalization layer, and a ReLU layer which can be trained and evaluated.
- **Analysis Tool**: User-mode sampling and tracing collection with VTune Profiler (version 2022 or newer)

​	**NOTE:**

- Starting with the 2020 release, Intel® VTune™ Amplifier has been renamed to Intel® VTune™ Profiler.
- Most recipes in the Intel® VTune™ Profiler Performance Analysis Cookbook are flexible. You can apply them to different versions of Intel® VTune™ Profiler. In some cases, minor adjustments may be required.
- Get the latest version of Intel® VTune™ Profiler:
  - From the [Intel® VTune™ Profiler product page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html).
  - Download the latest standalone package from the [Intel® oneAPI standalone components page](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler-download.html).
- **CPU**: 11th Gen Intel® Core(TM) i7-1165G7 @ 2.80GHz
- **Operating System**: Ubuntu Server 20.04.5 LTS

**Run VTune Profiler on a Python Application**

Let us start by running a [Hotspots analysis](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/basic-hotspots-analysis.html) on the Intel\_Extension\_For\_PyTorch\_Hello\_World.py ML application, without any change to the code. This analysis is a good starting point to identify the most time-consuming regions in the code.

In the command line, type:

`vtune -collect hotspots -knob sampling-mode=sw -knob enable-stack-collection=true -source-search-dir=path\_to\_src -search-dir /usr/bin/python3 -result-dir vtune\_hotspots\_results -- python3 Intel\_Extension\_For\_PyTorch\_Hello\_World.py`

Once the analysis completes, see the most active functions in the code by examining the **Top Hotspots** section in the **Summary** window.

![](Aspose.Words.9eb062de-7fb7-4a04-ad00-cd3eeeba07e1.001.png)

In this case, we see that the sched\_yield function consumes considerable CPU time. Excessive calls to this function can cause unnecessary context switches and result in a degradation of application performance.

Next, let us look at the top tasks in the application:

![](Aspose.Words.9eb062de-7fb7-4a04-ad00-cd3eeeba07e1.002.png)

Here we can see that the convolution task consumes the most processing time for this code.

While you can dig deeper by switching to the Bottom-up window, it may be challenging to isolate the most interesting regions for optimization. This is particularly true for larger applications because there may be a lot of model operators and functions in every layer of the code. Therefore, we will now add Intel® Instrumentation and Tracing Technology (ITT) APIs to generate results that are easier to interpret.

**Include ITT-Python APIs**

Let us now add Python\* bindings available in [ITT-Python](https://github.com/NERSC/itt-python) to the Intel® Instrumentation and Tracing Technology (ITT) APIs used by VTune Profiler. These bindings include user task labels to control data collection and some user task APIs (that can create and destroy task instances).

ITT-Python uses three types of APIs:

- Domain APIs
  - domain\_create(name)
- Task APIs
  - task\_begin(domain, name)
  - task\_end(domain)
- Anomaly Detection APIs
  - itt\_pt\_region\_create(name)
  - itt\_pt\_region\_begin(region)
  - itt\_pt\_region\_end(region)

The following example from [TensorFlow_HelloWorld.py](https://github.com/oneapi-src/oneAPI-samples/blob/master/AI-and-Analytics/Getting-Started-Samples/IntelTensorFlow_GettingStarted/TensorFlow_HelloWorld.py) calls the Domain and Task APIs in ITT-Python:

`itt.resume()`

`domain = itt.domain\_create("Example.Domain.Global")`

`itt.task\_begin(domain, "CreateTrainer")`

`for epoch in range(0, EPOCHNUM):`

`for step in range(0, BS\_TRAIN):`

`x\_batch = x\_data[step\*N:(step+1)\*N, :, :, :]`

`y\_batch = y\_data[step\*N:(step+1)\*N, :, :, :]`

`s.run(train, feed\_dict={x: x\_batch, y: y\_batch})`

`'''Compute and print loss. We pass Tensors containing the predicted and    true values of y, and the loss function returns a Tensor containing the loss.'''`

`print(epoch, s.run(loss,feed\_dict={x: x\_batch, y: y\_batch}))`

`itt.task\_end(domain)`

`itt.pause()`

Here is the sequence of operations:

1. Use itt.resume() API to resume the profiling just before the loop begins to execute.
1. Create an ITT domain (like Example.Domain.Global) for a majority of the ITT API calls.
1. Use the itt.task.begin() API to start the task. Label the task as CreateTrainer. This label appears in profiling results.
1. Use itt.task() API to end the task.
1. Use itt.pause() API to pause data collection.

Run Hotspots and Microarchitecture Exploration Analyses

Once you have modified your code, run the Hotspots analysis on the modified code.

`vtune -collect hotspots -start-paused -knob enable-stack-collection=true -knob sampling-mode=sw  -search-dir=/usr/bin/python3 -source-search-dir=path\_to\_src  -result-dir vtune\_data -- python3 TensorFlow\_HelloWorld.py`

This command uses the -start-paused parameter to profile only those code regions marked by ITT-Python APIs. Let us look at the results of the new Hotspots analysis. The **Top Hotspots** section displays hotspots in the code regions marked by ITT-Python APIs.

![](Aspose.Words.9eb062de-7fb7-4a04-ad00-cd3eeeba07e1.003.png)

Examine the most time-consuming ML primitives in the target code region. Focus on these primitives first to improve optimization. Using the ITT-APIs helps you identify those hotspots quickly which are more pertinent to ML primitives.

Next, look at the top tasks targeted by the ITT-Python APIs. Since you can use these APIs to limit profiling results to specific code regions, the ITT logical tasks in this section display information including:

- CPU time
  - Effective time
  - Spin time
  - Overhead time
- CPU utilization time
- Source code line level analysis

![](Aspose.Words.9eb062de-7fb7-4a04-ad00-cd3eeeba07e1.004.png)

The source line level profiling of the ML code reveals the source line breakdown of CPU time. In this example, the code spends 10.2% of the total execution time to train the model.

![](Aspose.Words.9eb062de-7fb7-4a04-ad00-cd3eeeba07e1.005.png)

To obtain a deeper understanding of application performance, let us now run the [Microarchitecture Exploration](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/current/general-exploration-analysis.html) analysis. In the command window, type:

`vtune -collect uarch-exploration -knob collect-memory-bandwidth=true  -source-search-dir=path\_to\_src -search-dir /usr/bin/python3 -result-dir vtune\_data\_tf\_uarch -- python3 TensorFlow\_HelloWorld.py`

Once the analysis completes, the **Bottom-up** window displays detailed profiling information for the tasks marked with ITT-Python APIs. We can see that the CreateTrainer task is front-end bound, which means that the front end is not supplying enough operations to the back end. Also, there is a high percentage of heavy-weight operations (those which need more than 2 µops).

![](Aspose.Words.9eb062de-7fb7-4a04-ad00-cd3eeeba07e1.006.png)

To focus your analysis on a smaller block of code, right click on one of the CreateTrainer tasks and enable filtering.

Add PyTorch\* ITT APIs (for PyTorch Framework only)

Just like ITT-Python APIs, you can also use [PyTorch* ITT APIs](https://pytorch.org/docs/stable/profiler.html#intel-instrumentation-and-tracing-technology-apis) with VTune Profiler. Use PyTorch ITT APIs to label the time span of individual PyTorch operators and get detailed analysis results for customized code regions. PyTorch 1.13 provides these versions of torch.profiler.itt APIs for use with VTune Profiler:

- is\_available()
- mark(msg)
- range\_push(msg)
- range\_pop()

Let us see how these APIs are used in a code snippet from [Intel_Extension_For_PyTorch_Hello_World.py](https://github.com/ONEAPI-SRC/ONEAPI-SAMPLES/BLOB/MASTER/AI-AND-ANALYTICS/GETTING-STARTED-SAMPLES/INTEL_EXTENSION_FOR_PYTORCH_GETTINGSTARTED/INTEL_EXTENSION_FOR_PYTORCH_HELLO_WORLD.PY).

`itt.resume()`

`with torch.autograd.profiler.emit\_itt():`

`torch.profiler.itt.range\_push('training')`

`model.train()`

`for batch\_index, (data, y\_ans) in enumerate(trainLoader):`

`data = data.to(memory\_format=torch.channels\_last)`

`optim.zero\_grad()`

`y = model(data)`

`loss = crite(y, y\_ans)`

`loss.backward()`

`optim.step()`

`torch.profiler.itt.range\_pop()`

`itt.pause()`

The above example features this sequence of operations:

1. Use the itt.resume() API to resume the profiling just before the loop begins to execute.
1. Use the torch.autograd.profiler.emit\_itt() API for a specific code region to be profiled.
1. Use the range\_push() API to push a range onto a stack of nested range spans. Mark it with a message ('*training'*).
1. Insert the code region of interest.
1. Use the range\_pop() API to pop a range from the stack of nested range spans.

Run Hotspots Analysis with PyTorch ITT APIs

Let us now run the Hotspots analysis for the code modified with PyTorch ITT APIs. In the command window, type:

`vtune -collect hotspots -start-paused -knob enable-stack-collection=true -knob sampling-mode=sw -search-dir=/usr/bin/python3 -source-search-dir=path\_to\_src  -result-dir vtune\_data\_torch\_profiler\_comb -- python3 Intel\_Extension\_For\_PyTorch\_Hello\_World.py`

Here are the top hotspots in our code region of interest:

![](Aspose.Words.9eb062de-7fb7-4a04-ad00-cd3eeeba07e1.007.png)

In the **Top Tasks** section of the **Summary**, we see the training task which was labeled using the ITT-API.

![](Aspose.Words.9eb062de-7fb7-4a04-ad00-cd3eeeba07e1.008.png)

When we examine the source line profiling of the PyTorch code, we see that the code spends 10.7% of the total execution time in backpropagation.

![](Aspose.Words.9eb062de-7fb7-4a04-ad00-cd3eeeba07e1.009.png)

Switch to the **Platform** window to see the timeline for the training task, which was marked using PyTorch ITT APIs.

![](Aspose.Words.9eb062de-7fb7-4a04-ad00-cd3eeeba07e1.010.png)

In the timeline, the main thread is python3(TID:125983) and it contains several smaller threads. Operator names that start with aten::batch\_norm, aten::native\_batch\_norm, aten::batch\_norm\_i are model operators.

From the Platform window, you can glean these details:

- CPU usage (for a specific time period) for every individual thread
- Start time
- Duration of user tasks and oneDNN primitives(Convolution, Reorder)
- Source lines for each task and primitive. Once the source file for a task/primitive is compiled with debug information, click on the task/primitive to see the source lines.
- Profiling results grouped by iteration number (when multiple iterations are available)

**Parent topic:** [Configuration Recipes](https://www.intel.com/content/www/us/en/docs/vtune-profiler/cookbook/2023-2/configuration-recipes.html)

See Also

[Intel® Optimization for TensorFlow*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/optimization-for-tensorflow.html#gs.v4v4k0)

[Intel® Optimization for PyTorch*](https://www.intel.com/content/www/us/en/developer/tools/oneapi/optimization-for-pytorch.html#gs.v4v8hv)

