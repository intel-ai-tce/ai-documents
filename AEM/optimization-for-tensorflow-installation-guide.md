
Intel® Optimization for TensorFlow\* Installation Guide
=======================================================

ID 767540

Updated 12/28/2022

Version Latest

Public

By

[PREETHI VENKATESH](https://community.intel.com/cipcp26785/plugins/custom/intel/intel/custom.userprofile?id=ml5gZ%2Bq2S%2FRuVuF5yVKOCw%3D%3D&iv=5440453458718028), [Jing Xu](https://community.intel.com/cipcp26785/plugins/custom/intel/intel/custom.userprofile?id=x%2BBLebNRNAv2RqmgOFwgig%3D%3D&iv=5832850724664211), [Hung-Ju Tsai](https://community.intel.com/cipcp26785/plugins/custom/intel/intel/custom.userprofile?id=gUEEIUHgEmoN5kvGGnYb%2Fg%3D%3D&iv=5872134306417728)

[TensorFlow\*](https://github.com/tensorflow/tensorflow) is a widely-used machine learning framework in the deep learning arena, demanding efficient utilization of computational resources. In order to take full advantage of Intel® architecture and to extract maximum performance, the TensorFlow framework has been optimized using oneAPI Deep Neural Network Library (oneDNN) primitives, a popular performance library for deep learning applications. For more information on the optimizations as well as performance data, see this blog post [TensorFlow\* Optimizations on Modern Intel® Architecture](/content/www/us/en/developer/articles/technical/tensorflow-optimizations-on-modern-intel-architecture.html).

Anaconda\* has now made it convenient for the AI community to enable high-performance-computing in TensorFlow. Starting from TensorFlow v1.9, Anaconda has and will continue to build TensorFlow using oneDNN primitives to deliver maximum performance in your CPU.

This install guide features several methods to obtain Intel Optimized TensorFlow including off-the-shelf packages or building one from source that are conveniently categorized into [Binaries](#binaries), [Docker Images](#docker_images), [Build from Source](#build_from_source). 

For more details of those releases, users could check [Release Notes of Intel Optimized TensorFlow](https://github.com/Intel-tensorflow/tensorflow/releases).

Now, Intel Optimization for Tensorflow is also available as part of [Intel® AI Analytics Toolkit](/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html). Download and Install to get separate conda environments optimized with Intel's latest AI accelerations. Code samples to help get started with are available [here](https://github.com/oneapi-src/oneAPI-samples/tree/master/AI-and-Analytics).

The oneAPI Deep Neural Network Library (oneDNN) optimizations are also now available in the official x86-64 TensorFlow after v2.5. The feature is off by default before v2.9, users can enable those CPU optimizations by setting the the environment variable **TF\_ENABLE\_ONEDNN\_OPTS=1** for the official x86-64 TensorFlow. **Since TensorFlow v2.9, the oneAPI Deep Neural Network Library (oneDNN) optimizations are enabled by default.** There is a comparison table between those two releases in the [additional information](#Additional Info) session.

Supported OS
------------

**For Linux\*:**

It supports most Linux distributions: like Ubuntu\*, Red Hat Enterprise Linux\*, SUSE Linux Enterprise Server\*, Fedora\*, CentOS\*, Debian\*, Amazon Linux 2\*, WSL 2, Rocky Linux\*, etc.

**For Windows\*:**

It supports Windows Server 2016\*, Windows Server 2019\*, Windows 10\*, Windows  11\*.

**NOTE: We recommend to use latest release of OS distributions to get better performance. It would show better performance in latest release than olders.**  

Supported Installation Options
------------------------------

**NOTE : Users can start with pip wheel installation from Intel Channel if no preference.**

| Catagory | Details |  Version |
| ----------- | ----------- | ----------- |
| Anaconda | **Linux:** Main Channel |  v2.10.0 |
|  | **Linux:** Intel AI Analytics Toolkit | v2.9.1 |
|  | **Windows**: Main Channel | v2.10.0 |
|  | **Windows**: Intel Channel | v2.8.0 |
|  | **MacOS:** Main Channel | v2.10.0 |
| _PIP Wheels_ | **Linux** | v2.11.0 |
|  | **Windows** | v2.11.0 |
| _Docker Containers_ | **Linux:** Intel containers | v2.9.1 |
|  | **Linux:** Google DL containers | v2.11 |
| _Build from source_ | **Linux** | NA |
| _Build from source_ | **Windows** | NA |


Installation Options
--------------------

### 1\. Binaries

#### Get Intel® Optimization for TensorFlow\* Pre-Built Images

####  [](#collapseCollapsible1678899283282) [Install the latest Intel® Optimization for TensorFlow\* from Anaconda\* Cloud](#collapseCollapsible1678899283282)

Available for Linux\*, Windows\*, MacOS\*

| **OS** | **TensorFlow\* version** | 
| -------- | -------- | 
| Linux\* | 2.10.0 | 
| Windows\*| 2.10.0 | 
| MacOS\* | 2.10.0 | 

  
Installation instructions:

If you don't have conda package manager, download and install [Anaconda](https://docs.anaconda.com/anaconda/install/)

Linux and MacOS

Open Anaconda prompt and use the following instruction

conda install tensorflow

In case your anaconda channel is not the highest priority channel by default(or you are not sure), use the following command to make sure you get the right TensorFlow with Intel optimizations

conda install tensorflow -c anaconda

Windows

Open Anaconda prompt and use the following instruction

conda install tensorflow-mkl

(or)

conda install tensorflow-mkl -c anaconda

Besides the install method described above, Intel Optimization for TensorFlow is distributed as wheels, docker images and conda package on Intel channel. Follow one of the installation procedures to get Intel-optimized TensorFlow.

Note: All binaries distributed by Intel were built against the TensorFlow version tags in a centOS container with gcc 4.8.5 and glibc 2.17 with the following compiler flags (shown below as passed to bazel\*)

\--cxxopt=-D\_GLIBCXX\_USE\_CXX11\_ABI=0 --copt=-march=corei7-avx --copt=-mtune=core-avx-i --copt=-O3 --copt=-Wformat --copt=-Wformat-security --copt=-fstack-protector --copt=-fPIC --copt=-fpic --linkopt=-znoexecstack --linkopt=-zrelro --linkopt=-znow --linkopt=-fstack-protector

**Note: please use the following instructions if you install TensorFlow\* v2.8 for missing python-flatbuffers module in TensorFlow\* v2.8.**

Linux and MacOS

conda install tensorflow python-flatbuffers 

Windows

conda install tensorflow-mkl python-flatbuffers 

####  [](#collapseCollapsible1678899283279) [Install the latest Intel® Optimization for TensorFlow\* from Intel Channel](#collapseCollapsible1678899283279)

Available for Linux\*, Windows\*


| **OS** | **TensorFlow\* version** | **Python Version** |
| -------- | -------- | ----------- |
| Linux\* | 2.9.1 | 3.9 |
| Windows\*| 2.8.0 | 3.7, 3.8, 3.9 and 3.10 |


Installation instructions:

Open Anaconda prompt and use the following instruction. 

conda install tensorflow -c intel

**Note: please use the following instructions if you install TensorFlow\* v2.8 for missing python-flatbuffers module in TensorFlow\* v2.8.**

conda install tensorflow python-flatbuffers -c intel

####  [](#collapseCollapsible1678899283278) [Install Intel® Optimization for TensorFlow\* from Intel® AI Analytics Toolkit](#collapseCollapsible1678899283278)

Available for Linux\*

TensorFlow\* version: 2.9.1

Installation instructions:

There are multiple options provided to download Intel® AI Analytics Toolkit, including Conda, online/offline installer, repositories and containers.

*   Installation via [repositories](https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/apt.html#apt)
*   Installation via [Anaconda](https://www.intel.com/content/www/us/en/develop/documentation/installation-guide-for-intel-oneapi-toolkits-linux/top/installation/install-using-package-managers/conda/install-intel-ai-analytics-toolkit-via-conda.html)

**All available download and installation guides can be found [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit-download.html?operatingsystem=linux)**

####  [](#collapseCollapsible1678899283276) [Install the Intel® Optimization for TensorFlow\* Wheel via PIP](#collapseCollapsible1678899283276)

Available for Linux\* and Windows\*

TensorFlow version: 2.11.0

**Installation instructions for 2.10 and later:**

Run the below instruction to install the wheel into an existing Python\* installation. Python versions supported are  3.7, 3.8, 3.9, 3.10

For Linux\* :

pip install intel-tensorflow==2.11.0

For Windows\*

pip install tensorflow-intel==2.11.0

If your machine has AVX512 instruction set supported please use the below packages for better performance.

pip install intel-tensorflow-avx512==2.11.0 # linux only

**Installation instructions for 2.9.1 and earlier:**

Run the below instruction to install the wheel into an existing Python\* installation. Python versions supported are  3.7, 3.8, 3.9, 3.10

pip install intel-tensorflow==2.9.1

If your machine has AVX512 instruction set supported please use the below packages for better performance.

pip install intel-tensorflow-avx512==2.9.1 # linux only

**Note: For TensorFlow versions 1.13, 1.14 and 1.15 with pip > 20.0, if you experience invalid wheel error, try to downgrade the pip version to < 20.0**

For e.g

python -m pip install --force-reinstall pip==19.0  
  
**Note: If your machine has AVX-512 instruction set supported, please download and install the wheel file with AVX-512 as minimum required instruction set from the table above, otherwise download and install the wheel without AVX-512. All Intel TensorFlow binaries are optimized with oneAPI Deep Neural Network Library (oneDNN), which will use the AVX2 or AVX512F FMA etc CPU instructions automatically in performance-critical operations based on the supported Instruction sets on your machine for both Windows and Linux OS.** 

**Note: If you ran into the following Warning on ISA above AVX2, please download and install the wheel file with AVX-512 as minimum required instruction set from the table above.**

I tensorflow/core/platform/cpu\_feature\_guard.cc:142\] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations: AVX2 AVX512F FMA To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

**Note: If you run a release with AVX-512 as minimum required instruction set on a machine without AVX-512 instruction set support, you will run into "Illegal instruction (core dumped)" error.**

**Note than for 1.14.0 install we have fixed a few vulnerabilities and the corrected versions can be installed using the below commands. We identified new CVE issues from curl and GCP support in the previous pypi package release, so we had to introduce a new set of fixed packages in PyPI**

**Available for Linux\* [here](https://pypi.org/project/intel-tensorflow/)**

####  [](#collapseCollapsible1678899283275) [Install the Official TensorFlow\* Wheel for running on Intel CPUs via PIP](#collapseCollapsible1678899283275)

Available for Linux\*

TensorFlow version: 2.11.0

Installation instructions:

Run the below instruction to install the wheel into an existing Python\* installation. Python versions supported are 3.7, 3.8, 3.9, 3.10

pip install tensorflow==2.11.0

The oneDNN CPU optimizations are enabled by default.

Please check [#Additional Info](#Additional Info) for differences between Intel® Optimization for TensorFlow\* and official TensorFlow\*.

### 2. Docker Images

#### Get Intel® Optimization for TensorFlow\* Docker Images

####  [](#collapseCollapsible1678899283273) [Google DL Containers](#collapseCollapsible1678899283273)

Starting version 1.14, Google released DL containers for TensorFlow on CPU optimized with oneDNN by default. The TensorFlow v1.x CPU container names are in the format "tf-cpu.", TensorFlow v2.x CPU container names are in the format "tf2-cpu." and support Python3. Below are sample commands to download the docker image locally and launch the container for TensorFlow 1.15 or TensorFlow 2.9. Please use one of the following commands at one time.

\# TensorFlow 1.15

docker run -d -p 8080:8080 -v /home:/home gcr.io/deeplearning-platform-release/tf-cpu.1-15

\# TensorFlow 2.11

docker run -d -p 8080:8080 -v /home:/home gcr.io/deeplearning-platform-release/tf2-cpu.2-11

This command will start the TensorFlow 1.15 or TensorFlow 2.8 with oneDNN enabled in detached mode, bind the running Jupyter server to port 8080 on the local machine, and mount local /home directory to /home in the container. The running JupyterLab instance can be accessed at localhost:8080.

To launch an interactive bash instance of the docker container, run one of the below commands.

\# TensorFlow 1.15

docker run -v /home:/home -it gcr.io/deeplearning-platform-release/tf-cpu.1-15 bash

\# TensorFlow 2.11

docker run -v /home:/home -it gcr.io/deeplearning-platform-release/tf2-cpu.2-11 bash

**Available Container Configurations**

You can find all supported docker tags/configurations [here](https://cloud.google.com/ai-platform/deep-learning-containers/docs/choosing-container).

####  [](#collapseCollapsible1678899283272) [Intel Containers at docker.com](#collapseCollapsible1678899283272)

Tensorflow Version: 2.9.1

These docker images are all published at [http://hub.docker.com](http://hub.docker.com) in [intel/intel-optimized-tensorflow](https://hub.docker.com/r/intel/intel-optimized-tensorflow) and [intel/intel-optimized-tensorflow-avx512](http://hub.docker.com/r/intel/intel-optimized-tensorflow-avx512/tags) namespaces and can be pulled with the following command:

\# intel-optimized-tensorflow

docker pull intel/intel-optimized-tensorflow

\# intel-optimized-tensorflow-avx512

docker pull intel/intel-optimized-tensorflow-avx512:latest

For example, to run the data science container directly, simply

\# intel-optimized-tensorflow

docker run -it -p 8888:8888 intel/intel-optimized-tensorflow

\# intel-optimized-tensorflow-avx512

docker run -it -p 8888:8888 intel/intel-optimized-tensorflow-avx512:latest

And then go to your browser on [http://localhost:8888/](http://localhost:8888/)

  
For those who want to navigate through the browser, follow the links:

*   For AVX as mimimum required instruction set: [https://hub.docker.com/r/intel/intel-optimized-tensorflow](https://hub.docker.com/r/intel/intel-optimized-tensorflow)
*   For AVX-512 as mimimum required instruction set: [https://hub.docker.com/r/intel/intel-optimized-tensorflow-avx512](https://hub.docker.com/r/intel/intel-optimized-tensorflow-avx512/tags)

**Available Container Configurations**

You can find all supported docker tags/configurations for [intel-optimized-tensorflow](https://hub.docker.com/r/intel/intel-optimized-tensorflow) and [intel-optimized-tensorflow-avx512](https://hub.docker.com/r/intel/intel-optimized-tensorflow-avx512/tags).

**To get the latest Release Notes on Intel® Optimization for TensorFlow\*, please refer this [article](/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html).**

**More containers for Intel® Optimization for TensorFlow\* can be found at the [Intel® oneContainer Portal](/content/www/us/en/developer/tools/containers/overview.html).**

### 3\. Build from Source

#### Build TensorFlow from Source with Intel oneAPI oneDNN library

####  [](#collapseCollapsible1678899283270) [Linux build](#collapseCollapsible1678899283270)

Building TensorFlow from source is not recommended. However, if instructions provided above do not work due to unsupported ISA, you can always build from source.

Building TensorFlow from source code requires Bazel installation, refer to the instructions here, [Installing Bazel](https://docs.bazel.build/versions/master/install.html#mac-os-x).

Installation instructions:

1.  Ensure numpy, keras-applications, keras-preprocessing, pip, six, wheel, mock packages are installed in the Python environment where TensorFlow is being built and installed.
2.  Clone the TensorFlow source code and checkout a branch of your preference
    *   git clone https://github.com/tensorflow/tensorflow
    *   git checkout r2.11
3.  Run "./configure" from the TensorFlow source directory
4.  Execute the following commands to create a pip package that can be used to install the optimized TensorFlow build.
    *   PATH can be changed to point to a specific version of GCC compiler:
        
        export PATH=/PATH//bin:$PATH
        
    *   LD\_LIBRARY\_PATH can also be to new:
        
        export LD\_LIBRARY\_PATH=/PATH//lib64:$LD\_LIBRARY\_PATH
        
    *   Set the compiler flags support by the GCC on your machine to build TensorFlow with oneDNN.
        
        bazel build --config=mkl -c opt --copt=-march=native //tensorflow/tools/pip\_package:build\_pip\_package
        
         
        *   If you would like to build the binary against certain hardware, ensure appropriate "march" and "mtune" flags are set. Refer the [gcc online docs](https://gcc.gnu.org/onlinedocs/) or [gcc x86-options](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html) to know the flags supported by your GCC version.
            
            bazel build --config=mkl --cxxopt=-D\_GLIBCXX\_USE\_CXX11\_ABI=0 --copt=-march=sandybridge --copt=-mtune=ivybridge --copt=-O3 //tensorflow/tools/pip\_package:build\_pip\_package
            
        *   Alternatively, if you would like to build the binary against certain instruction sets, set appropriate "Instruction sets" flags:
            
            bazel build --config=mkl -c opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mavx512f --copt=-mavx512pf --copt=-mavx512cd --copt=-mavx512er //tensorflow/tools/pip\_package:build\_pip\_package
            
            **Flags set above will add AVX, AVX2 and AVX512 instructions which will result in "illegal instruction" errors when you use older CPUs. If you want to build on older CPUs, set the instruction flags accordingly.**
            
        *   Users could enable additional oneDNN features by passing a "--copt=-Dxxx" build option.  For example, enable ITT\_TASKS feature from oneDNN by using below build instruction. User could refer to [oneDNN build options](https://oneapi-src.github.io/oneDNN/dev_guide_build_options.html) for more details.
            
            bazel build --config=mkl -c opt --copt=-march=native --copt=-DDNNL\_ENABLE\_ITT\_TASKS=True //tensorflow/tools/pip\_package:build\_pip\_package
5.  Install the optimized TensorFlow wheel
    *   bazel-bin/tensorflow/tools/pip\_package/build\_pip\_package ~/path\_to\_save\_wheel
    *   pip install --upgrade --user ~/path\_to\_save\_wheel/

####  [](#collapseCollapsible1678899283267) [Windows Build](#collapseCollapsible1678899283267)

**\* Prior to TensorFlow 2.3**

**Prerequisites**

Install the below Visual C++ 2015 build tools from [https://visualstudio.microsoft.com/vs/older-downloads/](https://visualstudio.microsoft.com/vs/older-downloads/)

*   Microsoft Visual C++ 2015 Redistributable Update 3
*   Microsoft Build Tools 2015 Update 3

**Installation**

1.  Refer to [Linux Section](/content/www/us/en/develop/articles/intel-optimization-for-tensorflow-installation-guide.html#linux_B_S) and follow Steps 1 through 3
2.  To build TensorFlow with oneDNN support, we need two additional steps.
    *   Link.exe on  Visual Studio 2015 causes the linker issue when /WHOLEARCHIVE switch is used. To overcome this issue, install the hotfix to your Visual C++ compiler available at [https://support.microsoft.com/en-us/help/4020481/fix-link-exe-crashes-with-a-fatal-lnk1000-error-when-you-use-wholearch](https://support.microsoft.com/en-us/help/4020481/fix-link-exe-crashes-with-a-fatal-lnk1000-error-when-you-use-wholearch)  
    *   Add a PATH environment variable to include MKL runtime lib location that will be created during the build process. The base download location can be specified in the bazel build command by using the --output\_base option, and the oneDNN libraries will then be downloaded into a directory relative to that base              
        *   set PATH=%PATH%;output\_dir\\external\\mkl\_windows\\lib

         3. Bazel build with the with "mkl" flag and the "output\_dir" to use the right mkl libs

             bazel --output\_base=output\_dir build --config=mkl --config=opt //tensorflow/tools/pip\_package:build\_pip\_package

          4. Install the optimized TensorFlow wheel

      bazel-bin\\tensorflow\\tools\\pip\_package\\build\_pip\_package C:\\temp\\path\_to\_save\_wheel

      pip install C:\\temp\\path\_to\_save\_wheel\\

**\* TensorFlow 2.3 and newer:**

**Prerequisites**

       Please follow the [Setup for Windows](https://www.tensorflow.org/install/source_windows#setup_for_windows) to prepare the build environment.

**Installation**

1.  Set the following environment variables:
    *        BAZEL\_SH: C:\\msys64/usr\\bin\\bash.exe
    *        BAZEL\_VS: C:\\Program Files (x86)\\Microsoft Visual Studio
    *        BAZEL\_VC: C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC
2.  Note: For [compile time reduction](https://github.com/Intel-tensorflow/tensorflow/blob/860ad1d719a6ad32da3bd551af39a95be0b2e8c3/configure.py#L1247), please set:
    *   set TF\_VC\_VERSION=16.6
    *   More details can be found [here](https://groups.google.com/a/tensorflow.org/d/topic/build/SsW98Eo7l3o/discussion).
3.  Add to the PATH environment variable to include
    *   python path, e.g. C:\\Program Files\\_Python-version_  # Python38
        
    *   oneDNN runtime lib location that will be created during the build process, e.g. D:\\output\_dir\\external\\mkl\\windows\\lib
        
    *   the Bazel path, e.g. C:\\Program Files\\_Bazel-version_  # Bazel-3.7.2
        
    *   MSYS2 path, e.g. C:\\msys64;C:\\msys64/usr\\bin
        
    *   Git path, e.g. C:\\Program Files\\Git\\cmd;C:\\Program Files\\Git/usr\\bin
        
        set PATH=%PATH%;C:\\Program Files\\Python38;D:\\output\_dir\\external\\mkl\_windows\\lib;C:\\Program Files\\Bazel-3.7.2;C:\\msys64;C:\\msys64/usr\\bin;C:\\Program Files\\Git\\cmd;C:\\Program Files\\Git/usr\\bin
4.  Download the TensorFlow source code, checkout the release branch, and configure the build:
    *   git clone https://github.com/Intel-tensorflow/tensorflow.git
    *   cd tensorflow
    *   git checkout _branch-name_ # r2.6, r2.7, etc.
    *   python ./configure.py
5.  Set the oneDNN output directory location outside TensorFlow home directory to avoid infinite symlink expansion error. Then add the path to the oneDNN output directory to the system PATH:   
    *   set OneDNN\_DIR=\\one\_dnn\_dir
        
    *   set PATH=%OneDNN\_DIR%;%PATH%
        
6.  Build TensorFlow from source with oneDNN. Navigate to the TensorFlow root directory tensorflow and run the following bazel command to build TensorFlow oneDNN from Source:
    *   bazel --output\_base=%OneDNN\_DIR% build --announce\_rc --config=opt --config=mkl --action\_env=PATH=""  --define=no\_tensorflow\_py\_deps=true  tensorflow/tools/pip\_package:build\_pip\_package
        

**Note: Based on [bazel issue #7026](https://github.com/bazelbuild/bazel/issues/7026) we set --action\_env=PATH=. Open cmd.exe, run echo %PATH% and copy the output to the value of --action\_env=PATH=. If found, please use single quotes with folder names of white space**s.

Additional Information
----------------------

####  [](#collapseCollapsible1678899283265) [Sanity Check](#collapseCollapsible1678899283265)

Once Intel-optimized TensorFlow is installed, running the below command must print "True" if oneDNN optimizations are present.

import tensorflow as tf

import os

def get\_mkl\_enabled\_flag():

    mkl\_enabled = False  
    major\_version = int(tf.\_\_version\_\_.split(".")\[0\])  
    minor\_version = int(tf.\_\_version\_\_.split(".")\[1\])  
    if major\_version >= 2:  
        if minor\_version < 5:  
            from tensorflow.python import \_pywrap\_util\_port  
        elif minor\_version >= 9:

            from tensorflow.python.util import \_pywrap\_util\_port  
            onednn\_enabled = int(os.environ.get('TF\_ENABLE\_ONEDNN\_OPTS', '1'))

        else:  
            from tensorflow.python.util import \_pywrap\_util\_port  
            onednn\_enabled = int(os.environ.get('TF\_ENABLE\_ONEDNN\_OPTS', '0'))  
        mkl\_enabled = \_pywrap\_util\_port.IsMklEnabled() or (onednn\_enabled == 1)  
    else:  
        mkl\_enabled = tf.pywrap\_tensorflow.IsMklEnabled()  
    return mkl\_enabled

print ("We are using Tensorflow version", tf.\_\_version\_\_)  
print("MKL enabled :", get\_mkl\_enabled\_flag())

####  [](#collapseCollapsible1678899283263) [Capture Verbose Log](#collapseCollapsible1678899283263)

For a deeper analysis of what oneDNN calls are being made under the hood, we have enabled a flag "ONEDNN\_VERBOSE" to catpure a log. Set the  environment variable ONEDNN\_VERBOSE=1 and run the Tensorflow script. You should see an output similar to below  printed on the console. This will ensure that the workload not only has MKL enabled but utlizes oneDNN calls underneath. 

<<picture here>>

Here is how to intepret the the log

Tensorflow optimizations are agnostic to the type of hardware, ISA supported, dtype, ops used in the workload, this data is used as a starting point by the Tensorflow optimization engineers to understand the perframance impact on your workload and help optimize further

####  [](#collapseCollapsible1678899283262) [Additional Capabilities and Known Issues](#collapseCollapsible1678899283262)

1.  For Intel® Optimization for TensorFlow\* for 4th Generation Intel® Xeon® Scalable processors, we publish one preview release for good performance. Please install it by   
     $ pip install intel-tensorflow==2.11.dev202242
2.  Intel-optimized TensorFlow enables oneDNN calls by default. **For v2.4 and previous version**, If at any point you wish to disable Intel MKL primitive calls, this can be disabled by setting TF\_DISABLE\_MKL flag to 1 before running your TensorFlow script.
    
    export TF\_DISABLE\_MKL=1       
    

           However, note that this flag will only disable oneDNN calls, but not MKL-ML calls. 

            Although oneDNN is responsible for most optimizations, certain ops are optimized by MKL-ML library, including matmul, transpose, etc. Disabling MKL-ML calls are not supported by TF\_DISABLE\_MKL flag at present and Intel is working with Google to add this functionality

       3. CPU affinity settings in Anaconda's TensorFlow: If oneDNN enabled TensorFlow is installed from the anaconda channel (not Intel channel), the "import tensorflow" command sets the KMP\_BLOCKTIME and OMP\_PROC\_BIND environment variables if not already set. However, these variables may have effects on other libraries such as Numpy/Scipy which use OpenMP or oneDNN. Alternatively, you can either set preferred values or unset them after importing TensorFlow. More details available in the TensorFlow GitHub [issue](https://github.com/tensorflow/tensorflow/issues/24172)

            import tensorflow # this sets KMP\_BLOCKTIME and OMP\_PROC\_BIND

            import os

            # delete the existing values

            del os.environ\['OMP\_PROC\_BIND'\]

           del os.environ\['KMP\_BLOCKTIME'\]

####  [](#collapseCollapsible1678899283260) [Differences between Intel Optimization for Tensorflow and official TensorFlow for running on Intel CPUs after v2.5](#collapseCollapsible1678899283260)

Although official TensorFlow has oneDNN optimizations by default, there are still some major differences between Intel Optimization for Tensorflow and official TensorFlow

**Here is a comparison table For TensorFlow v2.9 and later.**

 

**Intel Optimization for Tensorflow**

**official TensorFlow (Running on Intel CPUs)**

**oneDNN optimiziations**  

Enabled by default

Enabled by default

**OpenMP Optimizations**

Enabled by default

N/A. use eigen thread pool instead

**Layout Format** 

TensorFlow native layout format by default. **No oneDNN blocked format support.**

TensorFlow native layout format by default. **No oneDNN blocked format support.**

**int8 support from oneDNN**

Enabled by default

Enabled by default

**Here is a comparison table For TensorFlow v2.8.**

 

**Intel Optimization for Tensorflow**

**official TensorFlow (Running on Intel CPUs)**

**oneDNN optimiziations**  

Enabled by default

Enable by setting environment variable TF\_ENABLE\_ONEDNN\_OPTS=1 at runtime

**OpenMP Optimizations**

Enabled by default

N/A. use eigen thread pool instead

**Layout Format** 

TensorFlow native layout format by default. **No oneDNN blocked format support.**

TensorFlow native layout format by default. **No oneDNN blocked format support.**

**int8 support from oneDNN**

Enabled by default

Enable by setting environment variable TF\_ENABLE\_ONEDNN\_OPTS=1 at runtime

**Here is a comparison table For TensorFlow v2.6 and v2.7.**

 

**Intel Optimization for Tensorflow**

**official TensorFlow (Running on Intel CPUs)**

**oneDNN optimiziations**  

Enabled by default

Enable by setting environment variable TF\_ENABLE\_ONEDNN\_OPTS=1 at runtime

**OpenMP Optimizations**

Enabled by default

N/A. use eigen thread pool instead

**Layout Format** 

TensorFlow native layout format by default.  Enable oneDNN blocked format by setting  environment variable TF\_ENABLE\_MKL\_NATIVE\_FORMAT=0

TensorFlow native layout format by default.  Enable oneDNN blocked format by setting  environment variable TF\_ENABLE\_ONEDNN\_OPTS=1   and TF\_ENABLE\_MKL\_NATIVE\_FORMAT=0 

**int8 support from oneDNN**

Enabled by default

**Enable by setting environment variable TF\_ENABLE\_ONEDNN\_OPTS=1 at runtime**

**Here is a comparison table for TensorFlow v2.5.**

 

**Intel Optimization for Tensorflow**

**official TensorFlow (Running on Intel CPUs)**

**oneDNN optimiziations**  

Enabled by default

Enable by setting environment variable TF\_ENABLE\_ONEDNN\_OPTS=1 at runtime

**OpenMP Optimizations**

Enabled by default

N/A. use eigen thread pool instead

**Layout Format** 

TensorFlow native layout format by default.  Enable oneDNN blocked format by setting  environment variable TF\_ENABLE\_MKL\_NATIVE\_FORMAT=0 

TensorFlow native layout format by default.  Enable oneDNN blocked format by setting  environment variable TF\_ENABLE\_ONEDNN\_OPTS=1   and TF\_ENABLE\_MKL\_NATIVE\_FORMAT=0 

**int8 support from oneDNN**

Enabled by setting the env-variable TF\_ENABLE\_MKL\_NATIVE\_FORMAT=0

Not supported

####  [](#collapseCollapsible1678899283256) [Intel® Extension for TensorFlow](#collapseCollapsible1678899283256)

Intel has released [Intel® Extension for TensorFlow](https://github.com/intel/intel-extension-for-tensorflow) to support optimizations on Intel dGPU ( currently for Flex series)  and CPU.

**Please Note that the ITEX CPU release at this moment is an experimental feature, and users are strongly encouraged to continue using Intel optimizations for TensorFlow as directed in this install guide**

More info on ITEX can be accessed from these resources for **Intel dGPUs( Flex series)**

**Category**

**Links**

Official Doc

[Get Started Document](https://intel.github.io/intel-extension-for-tensorflow/latest/get_started.html)

Blog

[Accelerating TensorFlow on Intel Data Center GPU Flex Series](https://blog.tensorflow.org/2022/10/accelerating-tensorflow-on-intel-data-center-gpu-flex-series.html)

Blog

[Meet the Innovation of Intel AI Software: ITEX](https://www.intel.com/content/www/us/en/developer/articles/technical/innovation-of-ai-software-extension-tensorflow.html)

####  [](#collapseCollapsible1678899283254) [4th Generation Intel® Xeon® Scalable Processors](#collapseCollapsible1678899283254)

Optimizations for 4th Generation Intel® Xeon® Scalable processors start from TensorFlow\* 2.11.

Official x86-64 TensorFlow has the 4th Gen Xeon scalable processors optimizations but the [dev release of Intel Optimization for TensorFlow](https://pypi.org/project/intel-tensorflow/2.11.dev202242/) has most up-to-date optimizations.

Please follow below instructions to install the dev release of Intel Optimization for TensorFlow 2.11.

`conda create -n intel-tf python=3.8 -y conda activate intel-tf pip install intel-tensorflow==2.11.dev202242`

Support
-------

If you have further questions or need support on your workload optimization, Please submit your queries at the [TensorFlow GitHub issues](https://github.com/tensorflow/tensorflow/issues) with the label "comp:mkl" or the [Intel AI Frameworks forum](https://forums.intel.com/s/topic/0TO0P000000Pms4WAC/intel-optimized-ai-frameworks).

Useful Resources
----------------

**Category**

**Links**

Installation & releases

[Intel® AI Analytics Toolkit](/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit-download.html)

[TensorFlow in Anaconda](https://www.anaconda.com/tensorflow-in-anaconda/)

[Intel TensorFlow Installation Guide](/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html)

[Build TensorFlow from source on Windows](https://www.tensorflow.org/install/source_windows)

[Intel-optimized TensorFlow on AWS](https://aws.amazon.com/about-aws/whats-new/2018/11/tensorflow1_12_mms10_launch_deep_learning_ami/)

[Intel oneContainer](/content/www/us/en/developer/tools/containers/overview.html)

Performance

[General BKMs to maximize performance](/content/www/us/en/developer/articles/technical/maximize-tensorflow-performance-on-cpu-considerations-and-recommendations-for-inference.html)

[Topology specific BKMs and tutorials](https://github.com/IntelAI/models/tree/master/docs)

[Intel Model Zoo with pretrained models](https://github.com/IntelAI/models)

[Getting Started with AutoMixedPrecisionMkl](/content/www/us/en/developer/articles/guide/getting-started-with-automixedprecisionmkl.html)

[Optimize pre-trained model](/content/www/us/en/developer/articles/technical/optimize-tensorflow-pre-trained-model-inference.html)

[Improve TensorFlow Performance on AWS by oneDNN](https://www.intel.com/content/www/us/en/developer/articles/technical/improve-tensorflow-performance-on-aws-instances.html#gs.3nhs4u) 

1

#### Product and Performance Information

1

Performance varies by use, configuration and other factors. Learn more at [www.Intel.com/PerformanceIndex](http://www.intel.com/PerformanceIndex "Follow link").

Get Help

<link rel="stylesheet" href="/etc.clientlibs/settings/wcm/designs/ver/230524/intel/clientlibs/pages/get-help.min.css" type="text/css">

*   [Company Overview](/content/www/us/en/company-overview/company-overview.html)
*   [Contact Intel](/content/www/us/en/support/contact-intel.html)
*   [Newsroom](/content/www/us/en/newsroom/home.html)
*   [Investors](https://www.intc.com/)
*   [Careers](/content/www/us/en/jobs/jobs-at-intel.html)
*   [Corporate Responsibility](/content/www/us/en/corporate-responsibility/corporate-responsibility.html)
*   [Diversity & Inclusion](/content/www/us/en/diversity/diversity-at-intel.html)
*   [Public Policy](/content/www/us/en/company-overview/public-policy.html)

*   [](https://www.facebook.com/Intel)
*   [](https://twitter.com/intel)
*   [](https://www.linkedin.com/company/intel-corporation)
*   [](https://www.youtube.com/user/channelintel?sub_confirmation=1)
*   [](https://www.instagram.com/intel/)

*   © Intel Corporation
*   [Terms of Use](/content/www/us/en/legal/terms-of-use.html)
*   [\*Trademarks](/content/www/us/en/legal/trademarks.html)
*   [Cookies](/content/www/us/en/privacy/intel-cookie-notice.html)
*   [Privacy](/content/www/us/en/privacy/intel-privacy-notice.html)
*   [Supply Chain Transparency](/content/www/us/en/corporate-responsibility/statement-combating-modern-slavery.html)
*   [Site Map](/content/www/us/en/siteindex.html)
*   [Do Not Share My Personal Information](/#)
*   [Recycling](/content/www/us/en/products/docs/boards-kits/nuc/nuc-compute-stick-recycling-program.html)

Intel technologies may require enabled hardware, software or service activation. // No product or component can be absolutely secure. // Your costs and results may vary. // Performance varies by use, configuration and other factors. // See our complete legal [Notices and Disclaimers](https://edc.intel.com/content/www/us/en/products/performance/benchmarks/overview/#GUID-26B0C71C-25E9-477D-9007-52FCA56EE18C). // Intel is committed to respecting human rights and avoiding complicity in human rights abuses. See Intel’s [Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html). Intel’s products and software are intended only to be used in applications that do not cause or contribute to a violation of an internationally recognized human right.

[![Intel Footer Logo](/content/dam/logos/intel-footer-logo.svg "Intel Footer Logo")](/content/www/us/en/homepage.html "Intel Footer Logo")

<link rel="stylesheet" href="/etc.clientlibs/settings/wcm/designs/ver/230524/intel/clientlibs/pages/commons-page.min.css" type="text/css"><script src="/etc.clientlibs/settings/wcm/designs/ver/230524/intel/clientlibs/pages/commons-page.min.js" defer></script> <link rel="preload" href="/etc.clientlibs/settings/wcm/designs/ver/230524/intel/clientlibs/pages/contact-us.min.css" as="style"><link rel="stylesheet" href="/etc.clientlibs/settings/wcm/designs/ver/230524/intel/clientlibs/pages/contact-us.min.css" type="text/css"> <script>!function(){var e=setInterval(function(){"undefined"!=typeof $CQ&&($CQ(function(){CQ\_Analytics.SegmentMgr.loadSegments("/etc/segmentation"),CQ\_Analytics.ClientContextUtils.init("/etc/clientcontext/intel",window.location.pathname.substr(0,window.location.pathname.indexOf(".")))}),clearInterval(e))},100)}();</script> <link rel="preload" as="style" href="/etc.clientlibs/settings/wcm/designs/intel/us/en/css/resources/css/intel.rwd.override.css"/> <link rel="stylesheet" type="text/css" href="/etc.clientlibs/settings/wcm/designs/intel/us/en/css/resources/css/intel.rwd.override.css"/>
