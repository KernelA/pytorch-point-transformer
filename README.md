# Pytorch Point Transformer

[Original article](https://arxiv.org/abs/2012.09164)

## Results

### Modelnet 40

[WandDB project (can be unavailable in the future)](https://wandb.ai/kernela/pytorch-point-transformer-modelnet40)

[WandDB report (can be unavailable in the future)](https://api.wandb.ai/links/kernela/s2puzshc)

You can download a trained model from the [model registry](https://docs.wandb.ai/guides/models).

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
pip install -r ./requirements.torch.gpu.txt
pip install -r ./requirements.cpu.txt -r ./requirements.base.txt
```

### GPU:
```
pip install -r ./requirements.torch.gpu.txt
pip install -r ./requirements.gpu.txt -r ./requirements.base.txt
```

Only for `presentation.ipynb`
```
pip install -r requirements.add.txt
```

### Train on the simple shapes dataset

```
dvc repro pipelines/simple_shapes/dvc.yaml:train
```


### Train on the ModelNet10/40 dataset

```
dvc repro pipelines/modelnet10/dvc.yaml:train
```
or
```
dvc repro pipelines/modelnet40/dvc.yaml:train
```

### Train segmentation model on PartNet

1. [Download dataset here](https://shapenet.org/)
2. Run: `python ./preprocess_partnet.py --data_dir <data_dir> --out_dir <where_to_store_processed>`
3. Edit: [partnet.yaml](configs/datasets/partnet.yaml). Set `dataset_dir` to the `<where_to_store_processed>`
4. Run: `dvc repro pipelines/partnet/dvc.yaml:train`
