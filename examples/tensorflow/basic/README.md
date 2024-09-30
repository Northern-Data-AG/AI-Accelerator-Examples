# TensorFlow distributed training

In this tutorial, we will walk through setting up and running distributed training using TensorFlow on multiple GPUs. 
We'll cover the following scenarios:

1. [Single-GPU Training](#single-gpu-training)
2. [Multi-GPU Training on a Single Node](#multi-gpu-training-on-a-single-node)

This tutorial is based on the official TensorFlow [distributed training guide](https://www.tensorflow.org/guide/distributed_training) 
and examples from the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/?_gl=1*ytgvqi*_ga*NzExNDU0NTY5LjE3MjU5NTE5NDI.*_ga_092EL089CH*MTcyNTk1MTk0Mi4xLjEuMTcyNTk1MTk0Ny42MC4wLjA).

---

## Setup
### Clone the repo
Clone the sub-repo within this folder:
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

### Installation of required packages
You can run TensorFlow on your GPU device using two methods:

1. Install the required packages directly in your computing environment.
2. Use a predefined container environment (e.g., from [NVIDIA's NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow)).

#### Option 1: Install TensorFlow with GPU support locally
To install TensorFlow with CUDA support, run the following command:
```bash
python3 -m pip install tensorflow[and-cuda] tensorflow_datasets
```

#### Option 2: Use TensorFlow with GPU Support in a container
For this method, ensure that the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html) is installed on your system. Then, use the following commands to pull and run a TensorFlow container:
```bash
docker pull nvcr.io/nvidia/tensorflow:24.08-tf2-py3
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it nvcr.io/nvidia/tensorflow:24.08-tf2-py3 /bin/bash
```

---

## Single-GPU Training
This section focuses on using a single GPU for training in TensorFlow. Refer to the official TensorFlow guide on 
[how to use a GPU](https://www.tensorflow.org/guide/gpu) for a deeper dive into GPU memory management. For simplicity, this guide will not cover those details. 

### Automatic GPU Usage

When a TensorFlow operation is executed on a system with both CPU and GPU devices, it will automatically run on 
the GPU by default. If you want to use a specific GPU (other than GPU#0), you can manually specify this in TensorFlow.

### Specifying GPU Devices with tf.device()

By default, TensorFlow will prioritize the GPU for executing operations. If you need to run specific operations on 
a different GPU (or the CPU), you can manually specify the desired device using the with tf.device() method.

The code below demonstrates how to specify the device for executing operations:
```python
import tensorflow as tf

# Place tensors on the CPU
with tf.device('/CPU:0'):   # To choose a different GPU, use '/GPU:1'
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Run on GPU#0
c = tf.matmul(a, b)
```

For more advanced control over GPU resources, TensorFlow also provides APIs to limit memory usage or control how 
memory is allocated on GPUs. You can explore these in the [TensorFlow GPU guide](https://www.tensorflow.org/guide/gpu).

---

## Multi-GPU Training on a Single Node

This guide explains how to train a model on multiple GPUs using TensorFlow's `tf.distribute.Strategy` API. 
It focuses on the `tf.distribute.MirroredStrategy`, which enables data parallelism with synchronous updates.

For more details, refer to the [official TensorFlow distributed training tutorial](https://www.tensorflow.org/guide/distributed_training).

### How MirroredStrategy Works

`MirroredStrategy` replicates the model across all available GPUs (or a subset you define) and synchronizes 
the updates across these replicas using *Data Parallelism*. Each GPU, or replica, computes its mini-batch, 
and after all replicas finish, the average gradients are calculated and applied across all replicas. This 
process is known as an *AllReduce* operation, and by default, TensorFlow uses [NVIDIA NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html#allreduce) 
for this purpose.

You can also limit the number of GPUs used by setting environment variables `CUDA_VISIBLE_DEVICES` and 
`CUDA_DEVICE_ORDER` or explicitly defining the GPUs in the `MirroredStrategy` constructor.

### Setting Up MirroredStrategy

Here is how you create a `MirroredStrategy` instance in TensorFlow:
```python
import tensorflow as tf

# Create a MirroredStrategy instance
mirrored_strategy = tf.distribute.MirroredStrategy()  # You can optionally specify devices=["/gpu:0", "/gpu:1"] to limit GPUs
print('Number of devices used by this strategy: {}'.format(mirrored_strategy.num_replicas_in_sync))
```

This prints the number of replicas (GPUs) that will be used for distributed training.

### Training with MirroredStrategy

To use `MirroredStrategy` in your training process, you need to place your model creation and compilation 
inside the strategy's scope. TensorFlow automatically handles the distribution of variables created within 
the scope, converting them into *MirroredVariables*, which will be replicated across the GPUs.

Here's an example using the `mirrored_strategy.scope()` method with [Keras'](https://keras.io/api/) `model.fit()`:
```python
with mirrored_strategy.scope():
    # Define a simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,),
                              kernel_regularizer=tf.keras.regularizers.L2(1e-4))
    ])
    
    # Compile the model
    model.compile(loss='mse', optimizer='sgd')

# Create an example dataset
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(400).batch(80)

# Train the model
model.fit(dataset, epochs=5)

# Evaluate the model
model.evaluate(dataset)
```

### Important Notes
- **Batch Size**: Ensure that the batch size is divisible by the number of replicas (GPUs). This guarantees 
that each replica receives mini-batches of the same size, which is critical for synchronous updates.
  - In the above example, the dataset batch size is set to 80, which is divisible by 8, ensuring even 
distribution across 8 GPUs. Adjust this accordingly based on the number of GPUs in your setup.
