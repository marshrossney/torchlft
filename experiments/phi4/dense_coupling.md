---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Coupling Layer Models for Phi Four

```python
import torch
import matplotlib.pyplot as plt

from torchlft.scripts.train import parser, main
from torchlft.nflow.utils import get_jacobian, get_model_jacobian
from torchlft.nflow.layer import Layer

from plot import (
    plot_metrics,
    plot_layer_jacobians,
    plot_layer_log_det_jacobians,
)

plt.style.use("seaborn-v0_8-paper")

CUDA_AVAILABLE = torch.cuda.is_available()
```

<!-- #raw jupyter={"source_hidden": true} -->
_config_str = """
model:
  class_path: torchlft.models.scalar.DenseCouplingModel
  init_args:
    target:
      lattice_length: {L}
      β: {beta}
      λ: {lam}
    flow:
      - class_path: AffineCouplingBlock
        init_args:
          transform:
            symmetric: {symmetric}
          net:
            sizes: {sizes}
            activation: tanh
            bias: {bias}
          n_layers: {n_layers}
      - class_path: SplineCouplingBlock
        init_args:
          transform:
            n_bins: 8
            bounds: 5.0
          net:
            sizes: {sizes}
            activation: tanh
            bias: {bias}
          n_layers: 2
      - class_path: GlobalRescaling
        init_args:
          init_scale: 0.5
          frozen: true

    partitioning: checkerboard

train:
  n_steps: {n_steps}
  batch_size: {batch_size}
  init_lr: 0.001
  display_metrics: false

output: false
cuda: {cuda}
"""

_defaults = dict(
    L=8,
    beta=0.576,
    lam=0.5,
    sizes=[128],
    bias=False,
    symmetric=True,
    n_layers=8,
    n_steps=4000,
    batch_size=4096,
    cuda=CUDA_AVAILABLE,
)
   

def get_config(**kwargs):
    kwargs = _defaults | kwargs
    config = _config_str.format(**kwargs)
    return parser.parse_string(config)
<!-- #endraw -->

```python
_config_str = """
model:
  class_path: torchlft.models.scalar.DenseCouplingModel
  init_args:
    target:
      lattice_length: {L}
      β: {beta}
      λ: {lam}
    flow:
      to_free:
        m_sq: 0.25
      affine:
        transform:
          symmetric: {symmetric}
        net:
          sizes: {sizes}
          activation: tanh
          bias: {bias}
        n_layers: {n_layers}
      spline:
        transform:
          n_bins: 8
          bounds: 4.0
        net:
          sizes: {sizes}
          activation: tanh
          bias: true
        n_layers: 2
      rescale:
        init_scale: 1.0
        frozen: true

    partitioning: checkerboard

train:
  n_steps: {n_steps}
  batch_size: {batch_size}
  init_lr: 0.001
  display_metrics: false

output: false
cuda: {cuda}
"""

_defaults = dict(
    L=8,
    beta=0.576,
    lam=0.5,
    sizes=[128],
    bias=False,
    symmetric=True,
    n_layers=8,
    n_steps=4000,
    batch_size=4096,
    cuda=CUDA_AVAILABLE,
)
   

def get_config(**kwargs):
    kwargs = _defaults | kwargs
    config = _config_str.format(**kwargs)
    return parser.parse_string(config)
```

```python
# TEST
config = get_config()
print(parser.dump(config))

instantiated_config = parser.instantiate_classes(config)
untrained_model = instantiated_config.model
_ = untrained_model(1)
print(untrained_model)
```

```python
config = get_config()
trained_model, logger = main(config)

acceptance = float(logger.get_data()["acc"].mean(dim=1)[-1])
print(f"Final acceptance: {acceptance}")

_ = plot_metrics(logger)
```

```python
config = get_config(bias=False, symmetric=True)
trained_model, logger = main(config)

acceptance = float(logger.get_data()["acc"].mean(dim=1)[-1])
print(f"Final acceptance: {acceptance}")

_ = plot_metrics(logger)
```

```python
with torch.no_grad():
    φ, _ = trained_model.metropolis_sample(10000)
    
plt.hist(φ.flatten(), bins=35)
```

```python

```
