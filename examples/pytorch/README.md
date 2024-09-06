# Setup the development environment

All pytorch's example are run within a docker container that has all the relevant dependencies and packages to get you started quickly, without having to worry about dependency management.

First, you can build the container as `docker build -t pytorch-dev .` and launch it interactively `docker run -it --gpus all -v $(pwd):/workspace pytorch-dev`. The build command creates a Docker container with all necessary dependencies and the launching commands gives you an interactive environment that mounts your current directory as a volume. You can for example attach VSCode to your running container and start your own development within the container.

# Examples

## 1. Simple distributed training example
To test that distributed training works correctly in Pytorch and `torch.distributed` is setup correctly, you can run this [example](./basic/README.md). The tutorial follows step-by-step the PyTorch official [distributed training tutorial](https://pytorch.org/tutorials/beginner/ddp_series_intro.html?utm_source=distr_landing&utm_medium=ddp_series_intro) and its [source code](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series).

This tutorial provides a comprehensive guide to implementing distributed training in PyTorch. It is designed to help developers and researchers scale their deep learning models from a single GPU to multiple GPUs across multiple nodes. The tutorial covers:
1. Single-GPU training as a baseline
2. Multi-GPU training on a single machine using DistributedDataParallel (DDP)
3. Multi-GPU training using torchrun for simplified launching
4. Multi-node training for scaling across multiple machines

Furthermore, it provides Docker setup for consistent environments, a detailed explanations of key distributed training concepts, a step-by-step code modifications to enable distributed training, a series of examples of running scripts for different distributed configurations and tips for trivial scaling from single-GPU to multi-node setups without complex workload managers such as SLURM. The latter will be covered in a more detailed setup.

## 2. Distributed training example of minGPT
As a more realistic example, we will train the [minGPT](https://github.com/karpathy/minGPT) model following Andrej Karpathy's work. The [tutorial](./basic/README.md) leverages the PyTorch examples' adaptation of Karpathy's script. In this case, we provide a single-node training recipe, while the multi-node training run with SLURM as cluster manager will follow soon.