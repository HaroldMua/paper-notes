# Inference System

System for machine learning inference.

- [Blog](#Blog)


## Potential ideas
- Horizontal Autoscaling of pods
- 

## Benchmark
- Wanling Gao, Fei Tang, Jianfeng Zhan, et al. "AIBench: A Datacenter AI Benchmark Suite, BenchCouncil". [[Paper]](https://arxiv.org/pdf/2005.03459.pdf) [[Website]](https://www.benchcouncil.org/AIBench/index.html)
- BaiduBench: Benchmarking Deep Learning operations on different hardware. [[Github]](https://github.com/baidu-research/DeepBench#inference-benchmark)
- Reddi, Vijay Janapa, et al. "Mlperf inference benchmark." arXiv preprint arXiv:1911.02549 (2019). [[Paper]](https://arxiv.org/pdf/1911.02549.pdf) [[GitHub]](https://github.com/mlperf/inference)
- Bianco, Simone, et al. "Benchmark analysis of representative deep neural network architectures." IEEE Access 6 (2018): 64270-64277. [[Paper]](https://arxiv.org/abs/1810.00736)
- Almeida, Mario, et al. "EmBench: Quantifying Performance Variations of Deep Neural Networks across Modern Commodity Devices." The 3rd International Workshop on Deep Learning for Mobile Systems and Applications. 2019. [[Paper]](https://arxiv.org/pdf/1905.07346.pdf)

## Autoscaling
- Vertical Autoscaling of GPU Resources for Machine Learning in the Cloud. *Hyeon-Jun Jang*. **IEEE International Conference on Big Data**, 2020 [[paper]](https://ieeexplore.ieee.org/document/9378248) (Citations 0)
  - 指出，云中的资源自动缩放分为水平自动缩放和垂直自动缩放。水平自动缩放可自动缩放虚拟机（VM）的数量，而垂直自动缩放可缩放VM保留的资源量[[paper]](); 
  - 提出了一种垂直自动缩放算法，利用Lyapunov优化在预算限制内提高GPU资源的利用率。我们的算法处理GPU和CPU资源之间的相关性，只需要资源利用率信息来决定扩展GPU资源。
  - 没有开源代码，基于Linux cgroup实现的CPU资源虚拟化和基于 与Timegraph: Gpu scheduling for real-time multi-tasking environments类似思想实现的GPU虚拟化（提供运行时信息，推测与时间片模式相似），亦没有介绍技术细节；
  - 该论文的GPU虚拟化遵循（C，P）表示的周期性资源模型。对于每个时段P，容器可以使用时间C之前的GPU资源。如果容器因GPU的非抢占特性而超限，则将在下一时段为容器分配较少的配额。如果一个容器在一段时间内使用GPU资源的时间少于C，那么它将在下一个时间段内通过向容器分配更大的配额来进行补偿；
  - 上一点，相较于KubeShare的GPU共享方案，推测同是利用时间片模式。不如kubeshare。
  
- Container orchestration with cost-efficient autoscaling in cloud computing environments. *M Rodriguez*. 2018 [[paper]](https://arxiv.org/abs/1812.00300) (Citations 26)
  - 一种全面的容器资源管理方法：
    - 确定容器的初始放置位置
    - 根据集群的工作负载自动调整容器数量
    - 重调度机制，将新增的容器调度到负载较低的节点

- Jily: Cost-Aware AutoScaling of Heterogeneous GPU for DNN Inference in Public Cloud. *Zhaoxing Wang*. **IEEE International Conference on Performance, Computing and Communications (IPCCC)**, 2019 [[paper]](https://ieeexplore.ieee.org/abstract/document/8958770) (Citations 0)

- InferLine: Latency-Aware Provisioning and Scaling for Prediction Serving Pipelines. *Daniel Crankshaw*. **SoCC**, 2020 [[paper]](https://dl.acm.org/doi/abs/10.1145/3419111.3421285) (Citations 6)

## Serving
- DeepCPU: Serving RNN-based Deep Learning Models 10x Faster. *Minjia Zhang*. **ATC**, 2018 [[paper]](https://www.usenix.org/conference/atc18/presentation/zhang-minjia) (Citations 67)

- Accelerating Large Scale Deep Learning Inference through DeepCPU at Microsoft. *Minjia Zhang*. **OpML**, 2019 [[paper]](https://www.usenix.org/conference/opml19/presentation/zhang-minjia) (Citations 6)
  - 介绍性论文. Introduction内容可以借鉴

- Deep Learning Inference Service at Microsoft. *Jonathan Soifer*. **OpML**, 2019 [[paper]](https://www.usenix.org/conference/opml19/presentation/soifer) (Citations 10)
  - 描述了深度学习推理服务（Deep learning inference service, DLIS）的特点和作用，为什么需要DLIS;
  - 图1显示了DLIS及其关键组件的概述。Model Master（MM）是一个单例编排器，负责通过考虑模型需求和硬件资源，将模型容器智能地配置到一个或多个服务器上。模型服务器（MS）是服务器单元，可以有数千个。它们有两个角色：路由和模型执行。MS接收来自客户端的传入请求，并将其有效地路由到承载所请求模型实例的另一个MS。从路由服务器接收请求的MS随后以低延迟执行请求。
  - ![DLIS_Architecture.png](./imgs/DLIS_Architecture.png)
  - 描述了DLIS的所需组件，以及各组件(Intelligent model placement, low-latency model execution, efficient routing)的功能.个人工作重点在前两个;
  - Intelligent model placement
    - Model Placement
    - Diverse Hardware Management
  - Low-Latency Model Execution. Different levels of optimization are required to achieve low-latency serving. DLIS supports both system- and model-level optimizations. [[paper]](https://www.usenix.org/conference/atc18/presentation/zhang-minjia). 我们主要关注系统层次的优化。
    - Resource Isolation and Data Locality
    - Server-to-Model Communication

- INFaaS: Automated Model-less Inference Serving. *Francisco Romero*. **ATC**, 2021 [[paper]](https://www.usenix.org/system/files/atc21-romero.pdf) [[github]](https://github.com/stanford-mast/INFaaS) (Citations 2)
  - INFaaS generates model-variants and their performance-cost profiles on different hardware platforms.INFaaS tracks the dynamic status of variants (e.g., over-loaded or interfered) using a state machine, to efficiently select the right variant for each query to meet the applica-tion requirements. Finally, INFaaS combines VM-level (horizontal scaling) and model-level autoscaling to dynamically react to the changing application requirements and request patterns.
  - A model-variant is a version of a model defined by the following aspects: (a) model architecture (e.g., ResNet50, VGG16), (b) programming framework, (e.g., TensorFlow, PyTorch, Caffe2, MXNet), (c) model graph optimizers (e.g., TensorRT, Neuron, TVM, XLA [72]), (d) hyperparameters (e.g., optimizing for batch size of 1, 4, 8, or 16), and (e) hardware platforms (e.g., Haswell or Skylake CPUs, V100 or T4 GPUs, FPGA, and accelerators, such as Inferentia, TPU, Catapult, NPU). 
  - 同样地，将ML lifecycle划分为训练和推理两阶段。并指出推理服务系统面临的挑战：
    - Diverse application requirements. 准确性、时效性要求不同。
    - Heterogeneous execution environments. CPU/GPU/TPU等异构执行环境。
    - Diverse model-variants. 如TVM编译优化。
  - 这种论文，涉及复杂的工程实现(c++), 多种软硬因素（如上，model-variant由多种变量定义）,不适合个人研究者。

- Towards Designing a Self-Managed Machine Learning Inference Serving System in Public Cloud. *JR Gunasekaran*. 2020 [[paper]](https://arxiv.org/abs/2008.09491) (Citations 1)

- Deep Learning Inference in Facebook Data Centers: Characterization, Performance Optimizations and Hardware Implications. *Jongsoo Park*. 2018 [[paper]](https://arxiv.org/abs/1811.09886) (Citations 114)

- [Daniel Crankshaw](https://dancrankshaw.com/)

## Blog

- 算法平台在线服务体系的演进与实践，美团技术. [[link]](https://mp.weixin.qq.com/s/lLpqX9idaS_6FP5Gl7lqRw)
  - 需求
    - 支持多框架
    - 支持GPU和CPU
    - batch_size为1的推理无法充分利用GPU资源，要单卡运行多个服务（单容器多服务或者多容器多服务）
  - 要点
    - 进程、线程、协程中选择协程（Gevent协程库提供并发）
	- Flask作为WSGI Web框架
	- 每个容器运行一个在线服务进程（或者，每个容器运行一个服务进程，运行多个容器？），支持加载多个服务（模型），服务的数量由模型大小和显存大小决定
	- 计算机视觉模型大小普遍在小几百MB，一张8G的GPU可加载十几个服务
	- 模型管理仓库
	- HDFS作模型的存储中心
	- 采用Redis将结果缓存在内存
	
- 通用深度学习推理服务， 58同城. [[github]](https://github.com/wuba/dl_inference)
  - Tensorflow Serving可以直接读取SavedModel格式模型文件进行部署，支持gRPC和RESTful API调用
  - pytorch不提供服务化部署组件，需要用户自己开发实现。一种是用服务化框架进行封装，如使用Flask框架部署一个HTTP服务，编写API进行请求处理，API里调用pytorch推理函数；第二种是ONNX→onnx格式→Tensorflow模型→Tensorflow Serving 
  - 基于Seldon对pytorch进行封装提供RPC服务调用

- 面向大规模AI在线推理的可靠性设计, UCloud. [[lind]](https://mp.weixin.qq.com/s/Ehb2cRH549Wb29ErkyAR9w)

- 模型在线推系统的高可用设计, 京东数科. [[link]](https://blog.csdn.net/JDDTechTalk/article/details/108635914ew=1)

- online serving的工程难题，知乎. [[link]](https://zhuanlan.zhihu.com/p/77664408)
  - 评论有亮点， 如：
    - 在线Serving的挑战在于低延时，高吞吐，大批量：
      - 延时即E2E完成一个请求预测的时间，我认为可以通过缓存技术，计算加速(OpenBLAS/MKL/CUDA)，还有Approximate技术(没有实践过)来实现；
	  - 在保证单机TPS的前提下，吞吐量交给k8s来做；
	  - 认为这几个问题中大批量是最难的，当请求的batch size从上百到成千上万，E2E延时就是个大问题了，比较挫的方法时客户端加并行，不知道还有没有更好的办法

- 谷歌云使用GPU. [[link]](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)

- 谷歌云自动扩展机器学习预测. [[link]](https://cloud.google.com/blog/products/ai-machine-learning/scaling-machine-learning-predictions)







