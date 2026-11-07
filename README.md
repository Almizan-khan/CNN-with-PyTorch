# MNIST CNN · PyTorch

A compact image-classification experiment with a reproducible training split, checkpoint selection, confusion matrices, and CPU inference benchmarks.

[![checks](https://github.com/Almizan-khan/CNN-with-PyTorch/actions/workflows/checks.yml/badge.svg)](https://github.com/Almizan-khan/CNN-with-PyTorch/actions/workflows/checks.yml)


## What this demonstrates

- A two-convolution network implemented directly in PyTorch.
- Separate training, validation and official test data; checkpoint selection uses validation loss.
- Reloadable checkpoints, single-image inference, and regression tests for numerical and serialization behavior.

## Quick start

Use Python 3.11. Install a suitable PyTorch build for your CPU or CUDA environment, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python mnistcnn.py --epochs 5 --output outputs/mnist
python predict.py outputs/mnist/model.pt path/to/digit.png --invert
```

On Windows, activate the environment with `.venv\Scripts\activate`. The first training run downloads MNIST. Inference expects a centered digit, 28×28 grayscale, light foreground on a dark background. Use `--invert` for dark digits on white paper.

## Architecture and evaluation

`28×28 image → Conv(32) → ReLU/Pool → Conv(64) → ReLU/Pool → Dropout → Linear(128) → 10 classes`

The maintained trainer uses a seeded 55,000/5,000 training/validation split. It chooses the checkpoint with lowest validation loss, then evaluates the official 10,000-example test set once. The architecture and SGD settings are retained from the original experiment.

Outputs include `model.pt` and `metrics.json`: per-epoch losses, validation accuracy, final test accuracy, confusion matrix, error examples, seed, hyperparameters, versions and training time. These files are generated locally; checkpoints are not committed to Git.

## Benchmarks

```bash
python benchmark.py --output reports/benchmark.json
python mnistcnn.py --epochs 2 --train-limit 10000 --test-limit 2000 --output outputs/quick-check
```

The first command measures an untrained model's CPU forward latency on synthetic inputs, with warmup, median and p95 timing. It does not measure classification quality. The second command is explicitly a subset experiment, not a full MNIST score. The historical README's >98% claim has not been reproduced as part of this refresh and is not used as a verified result.

CI publishes its benchmark JSON as an Actions artifact. See [validation](VALIDATION.md) for test commands and [original experiment](notebooks/original_experiment.ipynb) for provenance.

## Limitations

MNIST is an introductory dataset. High accuracy here does not establish performance on photographs, handwriting outside the dataset, or production distribution shifts. The model is not calibrated for out-of-distribution inputs.

## References

- [MNIST](https://yann.lecun.com/exdb/mnist/)
- [PyTorch MNIST example](https://github.com/pytorch/examples/tree/main/mnist)
