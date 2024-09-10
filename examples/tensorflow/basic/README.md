# TensorFlow distributed training
This tutorial guides you through setting up and running distributed training using TensorFlow on multiple 
GPUs and nodes. We'll cover single-GPU training, multi-GPU training on a single node, and multi-node training.
The tutorial follows step-by-step the TensorFlow official [distributed training tutorial](https://www.tensorflow.org/guide/distributed_training) 
and the book [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 3rd Edition](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/?_gl=1*ytgvqi*_ga*NzExNDU0NTY5LjE3MjU5NTE5NDI.*_ga_092EL089CH*MTcyNTk1MTk0Mi4xLjEuMTcyNTk1MTk0Ny42MC4wLjA.).


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
Install TensorFlow with GPU support:
```bash
python3 -m pip install tensorflow[and-cuda] tensorflow_datasets
```


## Single-GPU training
TODO: `tf.strategy.OneDeviceStrategy`


## Multi-GPU training
This guide is based on the [official TensorFlow tutorial](https://www.tensorflow.org/guide/distributed_training) 
on training a model on multiple GPUs. It uses the `tf.distribute.Strategy` API to distribute training 
across multiple GPUs. We limit the strategy scope to `tf.distributed.MirroredStrategy`. This strategy uses 
Data Parallelism with synchronous updates, i.e. the model parameters are (by default) copied to all GPUs 
(replicas) of the server and updated as soon as all replicas finished computing their mini-batches. Those 
updates are done by calculating the average gradients over all gradients in that run. Such an operation is 
typically called *AllReduce* operation. By default, `tf.distributed.MirroredStrategy` uses [NVIDIA NCCL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html#allreduce) 
for that operation. You can also restrict your TensorFlow script so that it only uses a certain number of 
GPUs by either setting the environment variables `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` or defining 
the desired GPUs in the constructor of the strategy. Here is how you can create a `MirroredStrategy` instance:

```python
import tensorflow as tf

mirrored_strategy = tf.distribute.MirroredStrategy()  # Optionally include devices=["/gpu:0", "/gpu:1"] in the constructor to select only a subset of GPUs
```

After executing the above line, we can start the training by using the `mirrored_strategy.scope()` method 
in combination with [Keras'](https://keras.io/api/) `model.fit()` method. This is easily possible as the 
`tf.distributed.Strategy` classes are integrated into `tf.keras`. The only requirement is to move the 
creation and compilation part of the model inside the `mirrored_strategy.scope()` method. The variables 
created inside that scope are *MirroredVariables* and are getting distributed and executed on the specified 
replicas. Here is an example:

```python
with mirrored_strategy.scope():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,),
                          kernel_regularizer=tf.keras.regularizers.L2(1e-4))])
  model.compile(loss='mse', optimizer='sgd')

# Example dataset, please replace
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(400).batch(80)  # 80 % 8 == 0 -> Make sure to keep it modulo 0 should the number of GPUs (here 8) change!
model.fit(dataset, epochs=5)
model.evaluate(dataset)
```

**IMPORTANT**: Always make sure that the batch size is divisible by the number of replicas you are using! 
This is necessary to ensure that each replica receives mini-batches of the same size.

## Multi-Node training
TODO: `tf.strategy.MultiWorkerMirroredStrategy`
