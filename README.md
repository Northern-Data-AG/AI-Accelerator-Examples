# AI-Accelerator-Examples

A series of example to make the best use of Taiga Cloud infrastructure for the AI Accelerator program

The examples on this repository are organized by deliverable.
Current proposed examples showcase ways to train and serve ML/AI models.

With time we will improve this documentation to include ever more complex
scenarios.

# Getting started

On the following examples you might need one or more machines with minimum
requirements (of software and hardware). Check the example requirements for
more information.

## Deploy Node[s]

Deploy your virtual or baremetal machine via the
[Taiga Cloud](https://cloud-portal.northerndata.eu/) WebUI.

Our team provides up-to-date (latest) and binary frozen images.
Create a new instance of your desired node flavor and select the
latest image `ubuntu2204 ofed cudadrv cudatk ctk YYYYMMDD`

Login to your newly created machine.
```bash
ssh [-i mykeyfile] ubuntu@<my_machine_ip>
```

# Training
## PyTorch
Tutorials are available [here](./training/pytorch/README.md). In particular, they cover:

1. Hello-world of [distributed training](./training/pytorch/basic/README.md)
2. Distributed training of [minGPT](./training/pytorch/llm/README.md)

# Serving
## vLLM
