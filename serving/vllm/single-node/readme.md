# Single Node

## Requirements

For this tutorial the following hardware requirements are needed:
* 1+ GPU cards (nvidia)
* Some local storage (50GiB)

The following system binaryes will be used:
* Python 3.10.12
* pip 22.0.2

## Installation

Clone the repository.

```bash
mkdir ws -p && cd ws
git clone https://github.com/vllm-project/vllm.git && cd vllm
# git checkout 9da25a88aa35da4b5ad7da545e6189e08c5f52f4
```

Install required binaries to setup vLLM
```bash
sudo apt install python3-pip
```

Install vLLM required dependencies in a new isolated environment
```
virtualenv .venv --python 3.10
source .venv/bin/activate

pip install vllm
```

## Deploy

Run your model:
```bash
vllm serve bigcode/starcoder2-7b --dtype auto --api-key token-abc123
# vllm serve bigcode/starcoder2-15b --dtype auto --api-key token-abc123
```

Test your service with a request:
```python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)
completion = client.completions.create(
  model="bigcode/starcoder2-7b",
  prompt="# Function that does quicksort:",
  max_tokens=200
)
print(completion.choices[0].text)
```

Outputs:
``` python
 Takes array and a comparison function
# and re-orders the array
def quicksort(array, desc_order);

  # if left is out of range of array
  if array.length == 0 || left < 0
    return
  end

  # if right is out of range of array
  if array.length == 0 || right > array.length - 1
    return
  end

  if left == right
    return array;
  end

  # compute the partition around the pivot that
  # is returned by quicksort_partition
  pivot_item = quicksort_partition(array, left, right, comparison);
  if pivot_item != nil
    quicksort(array, left, pivot_item - 1, comparison);
    quicksort(array, pivot_item + 1, right, comparison);
  end

  return array;
end

# Firstarging function for the selection sort method
# Uses function argument to sort the array
```

Using curl:
```bash
curl -X GET http://localhost:8000/v1/models \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer token-abc123" \

curl -X POST http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer token-abc123" \
    -d '{
          "model": "bigcode/starcoder2-7b",
          "prompt": "# Function that does quicksort:",
          "temperature": 0,
          "max_tokens": 20
        }'
```

Outputs:
```json
{"id":"cmpl-00cae8b1251148cdb5833feb9242c966","object":"text_completion","created":1726564366,"model":"bigcode/starcoder2-7b","choices":[{"index":0,"text":"\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    else:\n        pivot = arr[0]\n        less = [i for i in arr[1:] if i <= pivot]\n        greater = [i for i in arr[1:] if i > pivot]\n        return quicksort(less) + [pivot] + quicksort(greater)\n\n# Function that does mergesort:\ndef mergesort(arr):\n    if len","logprobs":null,"finish_reason":"length","stop_reason":null,"prompt_logprobs":null}],"usage":{"prompt_tokens":7,"total_tokens":107,"completion_tokens":100}}
```

---

Let's create a systemd service to enable our application to start automatically
when it crashes or after a system reboot.
```bash
sudo tee /etc/systemd/system/vllm.service >/dev/null <<-EOF
[Unit]
Description=vLLM Server
Requires=network-online.target systemd-resolved.service
After=network-online.target systemd-resolved.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ws/vllm
ExecStart=/home/ubuntu/ws/vllm/.venv/bin/vllm serve bigcode/starcoder2-7b --dtype auto --api-key token-abc123
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo chmod 640 /etc/systemd/system/vllm.service

sudo systemctl daemon-reload
sudo systemctl enable vllm.service
sudo systemctl start vllm.service
```

Finaly, add a firewall rule to enable external access:
```bash
sudo ufw allow proto tcp from ::/0 to any port 8000
```

### SSL

In a real world scenario, we want to secure client-server connections.
vLLM will allow to add the necessary ssl keyfile, certificate and CA certs.
Check the following flags:
```bash
  --ssl-keyfile SSL_KEYFILE
                        The file path to the SSL key file
  --ssl-certfile SSL_CERTFILE
                        The file path to the SSL cert file
  --ssl-ca-certs SSL_CA_CERTS
                        The CA certificates file
  --ssl-cert-reqs SSL_CERT_REQS
                        Whether client certificate is required (see stdlib ssl
                        module's)
```

However this will require manual maintenance and awareness. We will be better
served using [Let's Encrypt](https://letsencrypt.org/) to do this for us.

## Shortcomings
* Using `vllm serve` can only use one api-key. A reverse-proxy together with
some sort of middleware is required to provide proper AuthZ/AuthN for
multi-user inference.
