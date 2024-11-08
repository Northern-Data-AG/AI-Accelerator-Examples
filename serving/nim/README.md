# NVIDIA NIMs

NVIDIA [NIMs](https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/) are self-contained inference-ready containers that include:
* Industry standard APIs for RESTfull [OpenAI compatible API](https://platform.openai.com/docs/api-reference/chat/create) requests.
* Optimized inference engines for LLMs by leveraging TensorRT optimizations.
* Enterprise level serving/runtime backed by [Triton Inference
Server](https://github.com/triton-inference-server).

NIMs easyness to deploy and adaptability to different server hardware makes it
the best starting point to start your serving jorney.

NIMs are a
[LICENSED](https://www.nvidia.com/en-us/launchpad/ai/generative-ai-inference-with-nim/) product and require authorization for commercial use.
Check all the available NIM artifacts from the [NVIDIA container
registry](https://catalog.ngc.nvidia.com/?filters=&orderBy=scoreDESC&query=label%3A%22NVIDIA+NIM%22&page=&pageSize=).


## How to get a NGCAPI key from TaigaCloud
TODO: How to obtain a NVAIE KEY From TAIGACLOUD


## Launch an NVIDIA NIM
1. Deploy Llamma3.1-70B on a [single node](./single-node/README.md)
