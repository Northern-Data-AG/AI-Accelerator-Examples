# Single Node

NVIDIA NIMs are the simplest and fastest way to deploy a production ready
foundation model. In this example we will deploy [Llama-3.1-70B-instruct](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama-3.1-70b-instruct).

## Requirements

For this tutorial the following hardware requirements are needed:
* 4+ GPU cards (NVIDIA) [A100 or better]
    * Please note that NIM profiles might impact requirements. Defined [tensor
    and pipeline
    parallelism](https://huggingface.co/docs/transformers/v4.15.0/parallelism#concepts) optimizations may require higher GPU count.
* Some local storage (100GiB)

## Installation

Create model directory.
```bash
export MODEL_DIRECTORY=/srv/models
sudo mkdir -p ${MODEL_DIRECTORY}
sudo chown $(id -u):$(id -u) ${MODEL_DIRECTORY}
```

With your [ngc personal key](https://org.ngc.nvidia.com/setup/personal-keys)
login into the NVIDIA container registry:
```bash
sudo docker login nvcr.io
> user: $oauthtoken
> pass: nvapi-XxXx
```

## Deploy

Test and launch the container.
> [!TIP]
> First deployment will always take more time as the model needs to be downloaded. Storing models in shared storage greatly improves deployment speed in multi-deployment.
```bash
export NGC_API_KEY=nvapi-XxXx
sudo -E USERID=$(id -u) docker run --rm --gpus '"device=1,2,3,4"' -e NGC_API_KEY -v ${MODEL_DIRECTORY}:/opt/nim/.cache nvcr.io/nim/meta/llama-3.1-70b-instruct:1.2.1
```

Your container should start with an output similar to:
```log
INFO 2024-09-27 12:59:39.857 ngc_profile.py:231] Running NIM without LoRA. Only looking for compatible profiles that do not support LoRA.
INFO 2024-09-27 12:59:39.857 ngc_profile.py:233] Detected 2 compatible profile(s).
INFO 2024-09-27 12:59:39.857 ngc_injector.py:152] Valid profile: 978c8d57db934f94121ca835f5aa93b292199900a22d6085add5ea733f92648c (tensorrt_llm-a100-bf16-tp4-throughput) on GPUs [0, 1, 2, 3]
INFO 2024-09-27 12:59:39.857 ngc_injector.py:152] Valid profile: 83542e200e12f0a019f9906fbe0ac915bfb34d6821d795598f297be8a663736f (vllm-bf16-tp4) on GPUs [0, 1, 2, 3]
INFO 2024-09-27 12:59:39.858 ngc_injector.py:206] Selected profile: 978c8d57db934f94121ca835f5aa93b292199900a22d6085add5ea733f92648c (tensorrt_llm-a100-bf16-tp4-throughput)
INFO 2024-09-27 12:59:39.860 ngc_injector.py:214] Profile metadata: feat_lora: false
INFO 2024-09-27 12:59:39.860 ngc_injector.py:214] Profile metadata: gpu: A100
INFO 2024-09-27 12:59:39.860 ngc_injector.py:214] Profile metadata: gpu_device: 20b2:10de
INFO 2024-09-27 12:59:39.860 ngc_injector.py:214] Profile metadata: llm_engine: tensorrt_llm
INFO 2024-09-27 12:59:39.860 ngc_injector.py:214] Profile metadata: pp: 1
INFO 2024-09-27 12:59:39.860 ngc_injector.py:214] Profile metadata: precision: bf16
INFO 2024-09-27 12:59:39.860 ngc_injector.py:214] Profile metadata: profile: throughput
INFO 2024-09-27 12:59:39.860 ngc_injector.py:214] Profile metadata: tp: 4
INFO 2024-09-27 12:59:39.860 ngc_injector.py:245] Preparing model workspace. This step might download additional files to run the model.
```

Using curl:
```bash
curl -X 'POST' \
  'http://[::1]:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [
      {
        "role":"user",
        "content":"Hello! How are you?"
      },
      {
        "role":"assistant",
        "content":"Hi! I am quite well, how can I help you today?"
      },
      {
        "role":"user",
        "content":"Can you write me a song?"
      }
    ],
    "top_p": 1,
    "n": 1,
    "max_tokens": 15,
    "stream": false,
    "frequency_penalty": 1.0,
    "stop": ["hello"]
  }'
```

Outputs:
```json
{
  "id": "chat-12396b07141c4ce8b095b2c974bed2d7",
  "object": "chat.completion",
  "created": 1727449389,
  "model": "meta/llama-3.1-70b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "I'd be happy to try. Do you have any ideas or specifications in"
      },
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 48,
    "total_tokens": 63,
    "completion_tokens": 15
  }
}
```

---

> [!NOTE]
> After downloading the model, the key is not required to be set in the contianer.

Let's create a systemd service to enable our application to start automatically
when it crashes or after a system reboot.
```bash
sudo tee /etc/systemd/system/nim.service >/dev/null <<-EOF
[Unit]
Description=NIM Server
Requires=network-online.target systemd-resolved.service
After=network-online.target systemd-resolved.service

[Service]
Type=simple
User=ubuntu
Environment="MODEL_DIRECTORY=/srv/models"
ExecStart=/usr/bin/docker run --rm --name nim_llm --gpus '"device=1,2,3,4"' -p 8000:8000 -v \${MODEL_DIRECTORY}:/opt/nim/.cache nvcr.io/nim/meta/llama-3.1-70b-instruct:1.2.1
ExecStop=/usr/bin/docker stop nim_llm
TimeoutStopSec=30
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo chmod 640 /etc/systemd/system/nim.service

sudo systemctl daemon-reload
sudo systemctl enable nim.service
sudo systemctl start nim.service
```

Finaly, add a firewall rule to enable external access:

> [!WARNING]
> Due to how this model serving was implemented in this tutorial, whis is an insecure way of exposing a service.

```bash
sudo ufw allow proto tcp from ::/0 to any port 8000
```

### Going Further

#TODO: Creating a ChatGPT like portal
