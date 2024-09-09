# Pytorch distributed training
This tutorial guides you through setting up and running distributed training using PyTorch on multiple GPUs and nodes. We'll cover single-GPU training, multi-GPU training on a single node, and multi-node training.

The tutorial follows step-by-step the PyTorch official [distributed training tutorial](https://pytorch.org/tutorials/beginner/ddp_series_intro.html?utm_source=distr_landing&utm_medium=ddp_series_intro) and its [source code](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series).

# Clone the repo

Clone the sub-repo within this folder
```bash
mkdir basic_parallelization
cd basic_parallelization/
git init
git remote add origin git@github.com:pytorch/examples.git
git fetch origin
git config core.sparseCheckout true
echo "distributed/ddp-tutorial-series/" >> .git/info/sparse-checkout
git pull origin main 
```

# Single-gpu training

Launch the previously built-container as `docker run -it --gpus all -v $(pwd):/workspace pytorch-dev` and, within the container, you can start with a simple single-GPU training script:
```bash
python distributed/ddp-tutorial-series/single_gpu.py 50 10
```
This script trains a basic Multi-Layer Perceptron (MLP) model on a single GPU. It's an excellent starting point to ensure everything is working correctly before moving to distributed training. The script uses standard PyTorch modules and utilities, including torch.nn for defining the model architecture and torch.optim for optimization.

The 2 flags sets the number of training epochs (50) and how often to save checkpoints (10).

# Multi-gpu training
Now that we have a working single-GPU script, let's scale up to multiple GPUs on a single machine. We'll use PyTorch's DistributedDataParallel (DDP) for this purpose. DDP is a module wrapper that enables efficient multi-GPU training by automatically handling the distribution of data and gradients across multiple GPUs.

1. Process Group Initialization: We use init_process_group to set up communication between processes. This function initializes the distributed environment and enables communication between different GPU processes.
2. DistributedDataParallel (DDP): This wrapper enables efficient multi-GPU training by automatically syncing gradients and model parameters across all GPUs. It ensures that the model remains consistent across all devices while allowing parallel computation.
3. DistributedSampler: This sampler ensures that each GPU gets different data samples during training. It partitions the dataset across the available GPUs, preventing redundant computations and ensuring each GPU processes a unique subset of the data.

Inside the running container, launch
```bash
python distributed/ddp-tutorial-series/multigpu.py 50 10
```

# Multi-gpu training with torchrun
`torchrun` simplifies the process of launching distributed training jobs. It handles setting up the distributed environment automatically.

Inside the running container, launch
```bash
torchrun --standalone --nproc_per_node=gpu distributed/ddp-tutorial-series/multigpu_torchrun.py 50 10
```

# Multi-node training

Just a few simple cosmetic changes from environmental variables for local and global ranks. To run, first launch the container with the extra option `--network=host`, as `docker run -it --gpus all --network=host -v $(pwd):/workspace pytorch-dev` on each one of the 2 nodes.

Then, inside the running container of the masternode, run the following:
```bash
torchrun \
--nproc_per_node=8 \
--nnodes=2 \
--node-rank=0 \
--rdzv-id=456 \
--rdzv-backend=c10d \
--rdzv-endpoint=[ipv6_address_master_node]:29603 \
basic_parallelization/distributed/ddp-tutorial-series/multinode.py --total_epochs 50 --save_every 10 --batch_size 32
```

while inside the running container of the workernode, run:
```bash
torchrun \
--nproc_per_node=8 \
--nnodes=2 \
--node-rank=1 \
--rdzv-id=456 \
--rdzv-backend=c10d \
--rdzv-endpoint=[ipv6_address_master_node]:29603 \
basic_parallelization/distributed/ddp-tutorial-series/multinode.py --total_epochs 50 --save_every 10 --batch_size 32
```

If working correctly and using a `world_size=16`, you should see a batch size of 4 per each node, as 
```bash
[GPU11] Epoch 47 | Batchsize: 32 | Steps: 4
```
