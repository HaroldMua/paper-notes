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

