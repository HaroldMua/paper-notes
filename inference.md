# Inference System

System for machine learning inference.

## Benchmark
- Wanling Gao, Fei Tang, Jianfeng Zhan, et al. "AIBench: A Datacenter AI Benchmark Suite, BenchCouncil". [[Paper]](https://arxiv.org/pdf/2005.03459.pdf) [[Website]](https://www.benchcouncil.org/AIBench/index.html)
- BaiduBench: Benchmarking Deep Learning operations on different hardware. [[Github]](https://github.com/baidu-research/DeepBench#inference-benchmark)
- Reddi, Vijay Janapa, et al. "Mlperf inference benchmark." arXiv preprint arXiv:1911.02549 (2019). [[Paper]](https://arxiv.org/pdf/1911.02549.pdf) [[GitHub]](https://github.com/mlperf/inference)
- Bianco, Simone, et al. "Benchmark analysis of representative deep neural network architectures." IEEE Access 6 (2018): 64270-64277. [[Paper]](https://arxiv.org/abs/1810.00736)
- Almeida, Mario, et al. "EmBench: Quantifying Performance Variations of Deep Neural Networks across Modern Commodity Devices." The 3rd International Workshop on Deep Learning for Mobile Systems and Applications. 2019. [[Paper]](https://arxiv.org/pdf/1905.07346.pdf)

## Autoscaling
- Vertical Autoscaling of GPU Resources for Machine Learning in the Cloud. *Hyeon-Jun Jang*. **IEEE International Conference on Big Data**, 2020 [[paper]](https://ieeexplore.ieee.org/document/9378248) (Citations 0) 
  - 提出了一种垂直自动缩放算法，利用Lyapunov优化在预算限制内提高GPU资源的利用率。我们的算法处理GPU和CPU资源之间的相关性，只需要资源利用率信息来决定扩展GPU资源。
  - 没有开源代码，基于Linux cgroup实现的CPU资源虚拟化和基于 与Timegraph: Gpu scheduling for real-time multi-tasking environments类似思想实现的GPU虚拟化（提供运行时信息，推测与时间片模式相似），亦没有介绍技术细节；
  - 该论文的GPU虚拟化遵循（C，P）表示的周期性资源模型。对于每个时段P，容器可以使用时间C之前的GPU资源。如果容器因GPU的非抢占特性而超限，则将在下一时段为容器分配较少的配额。如果一个容器在一段时间内使用GPU资源的时间少于C，那么它将在下一个时间段内通过向容器分配更大的配额来进行补偿；
  - 上一点，相较于KubeShare的GPU共享方案，推测同是利用时间片模式。不如kubeshare。

## Serving
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
