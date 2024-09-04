# NanoGPT Training

A simple example of distributed training of nanoGPT

# Develop with VSCode remotely
xyz

# Setup the development environment

Build the container:
```bash
docker build -t pytorch-dev .
```

Launch it interactively:
```bash
docker run -it --gpus all -v $(pwd):/workspace pytorch-dev
```

add the option `--network=host` when you run multi-node.

Attach vscode to the running container:
ctrl+shit+p: Dev Containers: Attach to running container