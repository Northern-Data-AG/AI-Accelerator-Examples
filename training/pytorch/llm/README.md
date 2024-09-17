# NanoGPT training

## Clone the example

This example makes use of the minGPT code by Karpathy. We can clone locally only the example following this:

```bash
mkdir pt-minGPT
cd pt-minGPT/
git init
git remote add origin git@github.com:pytorch/examples.git
git fetch origin
git config core.sparseCheckout true
echo "distributed/minGPT-ddp/mingpt" >> .git/info/sparse-checkout
git pull origin main 
```

which will clone the Pytorch's example files in the local folder called `pt-minGPT`.

## Run the training on a multi-gpu

Inside the proper folder `cd distributed/minGPT-ddp/mingpt/`, you can launch the previously built container
```bash
docker run -it --gpus all -v $(pwd):/workspace pytorch-dev`
```

and inside the container, run
```bash
torchrun --standalone --nproc_per_node=gpu main.py
```
