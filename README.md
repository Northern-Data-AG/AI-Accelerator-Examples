# AI-Accelerator-Examples
A series of example to make the best use of Taiga Cloud infrastructure for the AI Accelerator program

# Getting started

## PyTorch
To test that distributed training works correctly in Pytorch and `torch.distributed` is setup correctly, you can run this [example](./examples/pytorch/basic/README.md). The tutorial follows step-by-step the PyTorch official [distributed training tutorial](https://pytorch.org/tutorials/beginner/ddp_series_intro.html?utm_source=distr_landing&utm_medium=ddp_series_intro) and its [source code](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series).

This tutorial provides a comprehensive guide to implementing distributed training in PyTorch. It is designed to help developers and researchers scale their deep learning models from a single GPU to multiple GPUs across multiple nodes. The tutorial covers:
1. Single-GPU training as a baseline
2. Multi-GPU training on a single machine using DistributedDataParallel (DDP)
3. Multi-GPU training using torchrun for simplified launching
4. Multi-node training for scaling across multiple machines

Furthermore, it provides Docker setup for consistent environments, a detailed explanations of key distributed training concepts, a step-by-step code modifications to enable distributed training, a series of examples of running scripts for different distributed configurations and tips for trivial scaling from single-GPU to multi-node setups without complex workload managers such as SLURM. The latter will be covered in a more detailed setup.