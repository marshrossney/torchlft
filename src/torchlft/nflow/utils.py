from typing import TypeAlias

import torch

from torchlft.nflow.model import Model

Tensor: TypeAlias = torch.Tensor
Tensors: TypeAlias = tuple[Tensor, ...]


@torch.enable_grad()
def compute_grad_pullback(model: Model, inputs: Tensor):
    inputs.requires_grad_(True)
    inputs.grad = None

    outputs, log_det_jacobian = model.flow_forward(inputs)
    pullback_action = model.compute_target(outputs) - log_det_jacobian

    (gradient,) = torch.autograd.grad(
        outputs=pullback_action,
        inputs=inputs,
        grad_outputs=torch.ones_like(pullback_action),
    )

    inputs.requires_grad_(False)
    inputs.grad = None

    return gradient


@torch.no_grad()
def get_jacobian(transform, inputs: Tensor):
    def forward(input):
        output, *_ = transform(input.unsqueeze(0))
        output = output.squeeze(0)
        return output, output

    jac, outputs = torch.vmap(
        torch.func.jacrev(forward, argnums=0, has_aux=True)
    )(inputs)

    return jac, inputs, outputs


@torch.no_grad()
def get_model_jacobian(model, batch_size: int):
    inputs, _ = model.sample_base(batch_size)
    return get_jacobian(model.flow_forward, inputs)
