# Testing

Run the fast Go and subprocess-protocol suite with:

```sh
go test ./...
go test -race ./...
```

The trainer bridge tests use a local stub and do not download CIFAR or require a GPU. Real training is optional and requires the Python packages in `requirements-trainer.txt` plus a configured dataset/cache.
