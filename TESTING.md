# Testing

Run the fast Go and subprocess-protocol suite with:

```sh
go test ./...
go test -race ./...
```

The trainer bridge tests use a local stub and do not download CIFAR or require a GPU. Real training is optional and requires the Python packages in `requirements-trainer.txt` plus a configured dataset/cache.

## Trainer image

The default glibc-based container installs pinned packages from the official CPU wheel index and includes and `scripts/train.py`:

```sh
docker build -t nasgo:trainer .
docker run --rm --entrypoint python nasgo:trainer -c \
  'import torch, torchvision; print(torch.__version__)'
docker run --rm nasgo:trainer search --config /app/configs/smoke.yaml
```

Datasets are downloaded into `/app/data`; mount a persistent volume there. GPU use needs a CUDA-compatible derivative and host runtime. Proxy mode does not use PyTorch.
