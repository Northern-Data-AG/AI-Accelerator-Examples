# AI-Accelerator-Examples

A series of example to make the best use of Taiga Cloud infrastructure for the AI Accelerator program

The examples on this repository are organized by deliverable.
Current proposed examples showcase ways to train and serve ML/AI models.

With time we will improve this documentation to include ever more complex scenarios.

## Getting started

On the following examples you might need one or more machines with minimum requirements (of software and hardware).
Check the example requirements for more information.

For the AI Accelerator program, H100 cloud instances are pre-created for you and no deployment action is needed.
Login to your instance via
```bash
ssh [-i mykeyfile] ubuntu@<my_machine_ip>
```

In general, customers of Taiga Cloud could deploy virtual or baremetal instances via the [Taiga Cloud](https://cloud-portal.northerndata.eu/) WebUI.
Our team provides up-to-date (latest) and binary frozen images.
Create a new instance of your desired node flavor and select the latest image `ubuntu2204 ofed cudadrv fm cudatk ctk YYYYMMDD`

## Training

Pytorch tutorials are available [here](./training/pytorch/README.md). In particular, they cover:

1. Hello-world of [distributed training](./training/pytorch/basic/README.md)
2. Distributed training of [minGPT](./training/pytorch/llm/README.md)

TensorFlow tutorials are available [here](./training/tensorflow/basic/README.md).

## Serving

vLLM-based serving is available [here](./serving/vllm/readme.md).

Self-hosted NIM (requires using own NGC key with access to NIM, not provided as part of this program) is available [here](./serving/nim/README.md).
