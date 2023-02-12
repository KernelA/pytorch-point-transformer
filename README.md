# Pytorch Point Transformer

[Original article](https://arxiv.org/abs/2012.09164)

## Description

### Presentation

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelA/pytorch-point-transformer/blob/develop/presentation.ipynb)

### Colab training

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelA/pytorch-point-transformer/blob/develop/colab_training.ipynb)

[More info](https://github.com/phygitalism/3DML-Habr-paper)

## Requirements

1. Python 3.8 or higher.
2. CUDA 11.3 or higher

## How to run

### CPU (training will be slow)
```
pip install -r ./requirements.torch.cpu.txt
pip install -r ./requirements.cpu.txt -r ./requirements.base.txt
```

### GPU:
```
pip install -r ./requirements.torch.cpu.txt
pip install -r ./requirements.cpu.txt -r ./requirements.base.txt
```

Only for `presentation.ipynb`
```
pip install -r requirements.add.txt
```

### Train on the simple shapes dataset

```
dvc repro -s -f pipelines/simple_shapes/dvc.yaml:train
```


### Train on the ModelNet10 dataset

```
dvc repro -s -f pipelines/modelnet10/dvc.yaml:train
```
