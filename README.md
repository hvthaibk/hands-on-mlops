# Hands-on MLOps

MNIST classification using [`Weights & Biases`](https://wandb.ai/site)

## Setup

#### Create a conda environment (Python 3.10.12)
```
conda create -n mlops python=3.10
conda activate mlops
```

#### Install Python packages
```
pip install -r requirements.txt
```

PyTorch can also be installed using wheel files downloadable from
`https://download.pytorch.org/whl/torch/`.


#### Setup [`pre-commit`](https://pre-commit.com/) hooks
```
pre-commit install
pre-commit run --all-files
```
