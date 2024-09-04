# Pytorch distributed training
This tutorial guides you through setting up and running distributed training using PyTorch on multiple GPUs and nodes. We'll cover single-GPU training, multi-GPU training on a single node, and multi-node training.

The tutorial follows step-by-step the PyTorch official [distributed training tutorial](https://pytorch.org/tutorials/beginner/ddp_series_intro.html?utm_source=distr_landing&utm_medium=ddp_series_intro) and its [source code](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series).

# Setup the development environment

Let's set up our development environment using Docker. First, you can build the container as `docker build -t pytorch-dev .` and launch it interactively `docker run -it --gpus all -v $(pwd):/workspace pytorch-dev`. This creates a Docker container with all necessary dependencies and mounts your current directory as a volume.

# Single-gpu training

Let's start with a simple single-GPU training script:
```bash
python /workspace/basic_parallelization/singlegpu.py --total_epochs 50 --save_every 10 --batch_size 32 
```
This script trains a basic Multi-Layer Perceptron (MLP) model on a single GPU. It's an excellent starting point to ensure everything is working correctly before moving to distributed training. The script uses standard PyTorch modules and utilities, including torch.nn for defining the model architecture and torch.optim for optimization.

The --total_epochs flag sets the number of training epochs, --save_every determines how often to save checkpoints, and --batch_size sets the number of samples processed in each training step. These parameters can be adjusted based on your specific requirements and available computational resources.

# Multi-gpu training
Now that we have a working single-GPU script, let's scale up to multiple GPUs on a single machine. We'll use PyTorch's DistributedDataParallel (DDP) for this purpose. DDP is a module wrapper that enables efficient multi-GPU training by automatically handling the distribution of data and gradients across multiple GPUs.

1. Process Group Initialization: We use init_process_group to set up communication between processes. This function initializes the distributed environment and enables communication between different GPU processes.
2. DistributedDataParallel (DDP): This wrapper enables efficient multi-GPU training by automatically syncing gradients and model parameters across all GPUs. It ensures that the model remains consistent across all devices while allowing parallel computation.
3. DistributedSampler: This sampler ensures that each GPU gets different data samples during training. It partitions the dataset across the available GPUs, preventing redundant computations and ensuring each GPU processes a unique subset of the data.

- Import necessary modules:
```diff
+import torch.multiprocessing as mp
+from torch.utils.data.distributed import DistributedSampler
+from torch.nn.parallel import DistributedDataParallel as DDP
+from torch.distributed import init_process_group, destroy_process_group
+import os
```
These imports provide the necessary tools for distributed training in PyTorch.

- Set up the distributed environment:
```diff
+def ddp_setup(rank, world_size):
+    """
+    Args:
+        rank: Unique identifier of each process
+        world_size: Total number of processes
+    """
+    os.environ["MASTER_ADDR"] = "localhost"
+    os.environ["MASTER_PORT"] = "12355"
+    init_process_group(backend="nccl", rank=rank, world_size=world_size)
+    torch.cuda.set_device(rank)
```
This function initializes the distributed environment. It sets the master address and port for process communication, initializes the process group using the NCCL backend (optimized for NVIDIA GPUs), and sets the current GPU device.

- Modify the `Trainer` class to use DDP:
```diff
 class Trainer:
     def __init__(
         train_data: DataLoader,
         optimizer: torch.optim.Optimizer,
         gpu_id: int,
         save_every: int
     ) -> None:
         self.gpu_id = gpu_id
         self.model = model.to(gpu_id)
         self.train_data = train_data
         self.optimizer = optimizer
         self.save_every = save_every
+        self.model = DDP(model, device_ids=[gpu_id])
```
This line wraps our model in DistributedDataParallel, which handles the distribution of the model across multiple GPUs.

- Set epoch in the data sampler:
```diff
     def _run_epoch(self, epoch):
         b_sz = len(next(iter(self.train_data))[0])
         print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
+        self.train_data.sampler.set_epoch(epoch)
         for source, targets in self.train_data:
             source = source.to(self.gpu_id)
             targets = targets.to(self.gpu_id)
             self._run_batch(source, targets)
```

- Access the model via module when saving checkpoints:
```diff
def _save_checkpoint(self, epoch):
-        ckp = self.model.state_dict()
+        ckp = self.model.module.state_dict()
    PATH = "checkpoint.pt"
    torch.save(ckp, PATH)
    print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
```

- Save checkpoints only for one GPU:
```diff
def train(self, max_epochs: int):
    for epoch in range(max_epochs):
        self._run_epoch(epoch)
-            if epoch % self.save_every == 0:
+            if self.gpu_id == 0 and epoch % self.save_every == 0:
            self._save_checkpoint(epoch)
```

- Use DistributedSampler in the data loader:
```diff
def prepare_dataloader(dataset: Dataset, batch_size: int):
         dataset,
         batch_size=batch_size,
         pin_memory=True,
-        shuffle=True
+        shuffle=False,
+        sampler=DistributedSampler(dataset)
     )
```
 
- Modify the main function to use DDP:
```diff
-def main(device, total_epochs, save_every, batch_size):
+def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
+    ddp_setup(rank, world_size)
     dataset, model, optimizer = load_train_objs()
     train_data = prepare_dataloader(dataset, batch_size)
-    trainer = Trainer(model, train_data, optimizer, device, save_every)
+    trainer = Trainer(model, train_data, optimizer, rank, save_every)
     trainer.train(total_epochs)
+    destroy_process_group()
``` 

- Launch the script using mp.spawn:
```diff
 if __name__ == "__main__":
     args = parser.parse_args()    
-    device = 0  # shorthand for cuda:0
-    main(device, args.total_epochs, args.save_every, args.batch_size)
+    world_size = torch.cuda.device_count()
+    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
```

Inside the running container, launch
```bash
python /workspace/basic_parallelization/multigpu.py --total_epochs 50 --save_every 10 --batch_size 32 
```

# Multi-gpu training with torchrun
`torchrun` simplifies the process of launching distributed training jobs. It handles setting up the distributed environment automatically.

- Torchrun handles itself master address and port, as well as rank and world_size:
```diff
-def ddp_setup(rank, world_size):
-    """
-    Args:
-        rank: Unique identifier of each process
-        world_size: Total number of processes
-    """
-    os.environ["MASTER_ADDR"] = "localhost"
-    os.environ["MASTER_PORT"] = "12355"
-    init_process_group(backend="nccl", rank=rank, world_size=world_size)
-    torch.cuda.set_device(rank)
+def ddp_setup():
+    init_process_group(backend="nccl")
+
```

- Change `Trainer` class to let torchrun handle gpu_id, add local_rank environmental vairable and add fault-tolerant checkpoint loading with new method `_load_snapshot`:
```diff
 class Trainer:
     def __init__(
         model: torch.nn.Module,
         train_data: DataLoader,
         optimizer: torch.optim.Optimizer,
-        gpu_id: int,
         save_every: int,
+        snapshot_path: str,
     ) -> None:
-        self.gpu_id = gpu_id
-        self.model = model.to(gpu_id)
+        self.gpu_id = int(os.environ["LOCAL_RANK"])
+        self.model = model.to(self.gpu_id)
         self.train_data = train_data
         self.optimizer = optimizer
         self.save_every = save_every
-        self.model = DDP(model, device_ids=[gpu_id])
+        self.epochs_run = 0
+        if os.path.exists(snapshot_path):
+            print(f"Loading checkpoint from {snapshot_path}") 
+            self._load_snapshot(snapshot_path)
+        self.model = DDP(self.model, device_ids=[self.gpu_id])
+
+    def _load_snapshot(self, snapshot_path):
+        snapshot = torch.load(snapshot_path)
+        self.model.load_state_dict(snapshot["MODEL_STATE"])
+        self.epochs_run = snapshot["EPOCHS_RUN"]
+        print(f"Resuming training from snapshot at epoch {self.epochs_run}")
```

- Refactor checkpoint saving as model state_dict, adding epoch run for re-loading from checkpoint and call saving method.
```diff
-    def _save_checkpoint(self, epoch):
-        ckp = self.model.module.state_dict()
-        PATH = "checkpoint.pt"
-        torch.save(ckp, PATH)
+    def _save_snapshot(self, epoch):
+        snapshot = {}
+        snapshot["MODEL_STATE"] = self.model.module.state_dict()
+        snapshot["EPOCHS_RUN"] = epoch
+        PATH = "snapshot.pt"
+        torch.save(snapshot, PATH)
         print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
```

- Refactor training logic to restart from epochs_run if checkpoint exists
```diff
     def train(self, max_epochs: int):
-        for epoch in range(max_epochs):
+        for epoch in range(self.epochs_run, max_epochs):
             self._run_epoch(epoch)
             if self.gpu_id == 0 and epoch % self.save_every == 0:
-                self._save_checkpoint(epoch)
+                self._save_snapshot(epoch)
``` 

- Refactor main as rank is not needed anymore (loaded as environmental variable)
```diff
-def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
-    ddp_setup(rank, world_size)
+def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
+    ddp_setup()
     dataset, model, optimizer = load_train_objs()
     train_data = prepare_dataloader(dataset, batch_size)
-    trainer = Trainer(model, train_data, optimizer, rank, save_every)
+    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
     trainer.train(total_epochs)
     destroy_process_group()
```

- Call to main does not neet mp.spawn as torchrun takes care of launching:
```diff
if __name__ == "__main__":
     args = parser.parse_args()
     
     world_size = torch.cuda.device_count()
-    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
+    main(args.save_every, args.total_epochs, args.batch_size)
```

Inside the running container, launch
```bash
torchrun --standalone --nproc_per_node=gpu /workspace/basic_parallelization/multigpu_torchrun.py --total_epochs 50 --save_every 10 --batch_size 32
```

# Multi-node training

- Just a few simple cosmetic changes from environmental variables for local and global ranks
```diff
class Trainer:
         save_every: int,
         snapshot_path: str,
     ) -> None:
-        self.gpu_id = int(os.environ["LOCAL_RANK"])
-        self.model = model.to(self.gpu_id)
+        self.local_rank = int(os.environ["LOCAL_RANK"])
+        self.global_rank = int(os.environ["RANK"])
+        self.model = model.to(self.local_rank)
         self.train_data = train_data
         self.optimizer = optimizer
         self.save_every = save_every
```

- DDP call
```diff
         if os.path.exists(snapshot_path):
             print(f"Loading checkpoint from {snapshot_path}") 
             self._load_snapshot(snapshot_path)
-        self.model = DDP(self.model, device_ids=[self.gpu_id])
+        self.model = DDP(self.model, device_ids=[self.local_rank])
```

- Epoch run call print only from global_rank
```diff
     def _run_epoch(self, epoch):
         b_sz = len(next(iter(self.train_data))[0])
-        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
+        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
         self.train_data.sampler.set_epoch(epoch)
         for source, targets in self.train_data:
-            source = source.to(self.gpu_id)
-            targets = targets.to(self.gpu_id)
+            source = source.to(self.local_rank)
+            targets = targets.to(self.local_rank)
             self._run_batch(source, targets)
```

- Saving snapshots from global_rank=0 only:
```diff
     def _save_snapshot(self, epoch):
     def train(self, max_epochs: int):
         for epoch in range(self.epochs_run, max_epochs):
             self._run_epoch(epoch)
-            if self.gpu_id == 0 and epoch % self.save_every == 0:
+            if self.local_rank == 0 and epoch % self.save_every == 0:
                 self._save_snapshot(epoch)
```

To run, first launch the container with the extra option `--network=host`, as `docker run -it --gpus all --network=host -v $(pwd):/workspace pytorch-dev` on each one of the 2 nodes.

Then, inside the running container of the masternode, run the following:
```bash
torchrun \
--nproc_per_node=8 \
--nnodes=2 \
--node-rank=0 \
--rdzv-id=456 \
--rdzv-backend=c10d \
--rdzv-endpoint=[ipv6_address_master_node]:29603 \
basic_parallelization/multinode.py --total_epochs 50 --save_every 10 --batch_size 32
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
basic_parallelization/multinode.py --total_epochs 50 --save_every 10 --batch_size 32
```

If working correctly and using a `world_size=16`, you should see a batch size of 4 per each node, as 
```bash
[GPU11] Epoch 47 | Batchsize: 32 | Steps: 4
```