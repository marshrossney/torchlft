from math import sqrt

import torch
import matplotlib.pyplot as plt

from torchlft.nflow.layer import Layer, Composition
from torchlft.nflow.model import Model
from torchlft.nflow.utils import get_jacobian, get_model_jacobian

plt.style.use("seaborn-v0_8-paper")


def plot_metrics(logger, figsize: tuple[int, int] = (6, 4)):
    metrics = logger.get_data()

    steps = metrics["steps"]
    kl_div = -metrics["mlw"]
    one_minus_ess = 1 - metrics["ess"]
    one_minus_acc = 1 - metrics["acc"]
    var_logw = metrics["vlw"]

    def _plot(ax, tensor, colour):
        q = torch.tensor([0.25, 0.75], dtype=tensor.dtype)
        fb = ax.fill_between(
            steps, *tensor.quantile(q, dim=1), color=colour, alpha=0.5
        )
        (l,) = ax.plot(
            steps, tensor.quantile(0.5, dim=1), color=colour, linestyle="--"
        )
        return (fb, l)

    fig, ax = plt.subplots(figsize=figsize)

    handles, labels = [], []

    handle = _plot(ax, kl_div, "tab:blue")
    handles.append(handle)
    labels.append("KL Divergence")

    handle = _plot(ax, one_minus_acc, "tab:orange")
    handles.append(handle)
    labels.append("1 - Acceptance")

    handle = _plot(ax, one_minus_ess, "tab:green")
    handles.append(handle)
    labels.append("1 - ESS")

    handle = _plot(ax, var_logw, "tab:red")
    handles.append(handle)
    labels.append(r"Variance of $\log (p_\theta / p^\ast)$")

    ax.set_yscale("log")
    ax.set_ylim(None, 10)

    ax.set_xlabel("Training step")

    ax.legend(handles, labels)

    fig.tight_layout()

    return fig


def plot_layer_jacobians(model):
    layers = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, Layer)
    ]
    layers.append(("full model", model.flow_forward))

    inputs, _ = model.sample_base(1)

    for label, layer in layers:
        jac, _, outputs = get_jacobian(layer, inputs)

        D = int(sqrt(jac.numel()))
        jac = jac.view(D, D)

        fig, ax = plt.subplots()
        ax.set_axis_off()
        ax.imshow(jac)
        ax.set_title(label)

        yield fig

        inputs = outputs


def _plot_layer_ldj(model, ax, batch_size):
    layers = [
        (name, mod)
        for name, mod in model.named_modules()
        if (isinstance(mod, Layer) and not isinstance(mod, Composition))
    ]

    inputs, _ = model.sample_base(batch_size)
    ldjs = []

    with torch.no_grad():
        for name, layer in layers:
            outputs, ldj = layer(inputs)
            inputs = outputs
            ldjs.append(ldj.squeeze(1))

    ldjs = torch.stack(ldjs).cumsum(dim=0)
    depth = list(range(1, len(ldjs) + 1))

    iqr = ax.fill_between(
        depth,
        ldjs.quantile(0.25, dim=1),
        ldjs.quantile(0.75, dim=1),
        alpha=0.5,
    )
    (mline,) = ax.plot(depth, ldjs.quantile(0.5, dim=1), linestyle="-")

    return (iqr, mline)


def plot_layer_log_det_jacobians(
    models: dict[str, Model] | Model, batch_size: int = 64
):
    # Allow single model
    if isinstance(models, Model):
        models = {"Model": models}

    assert (
        len(set([model.target.lattice_length for model in models.values()]))
        == 1
    )
    assert len(set([model.target.m_sq for model in models.values()])) == 1

    fig, ax = plt.subplots()
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cumulative log det Jacobian")

    correct = list(models.values())[0].target.cholesky.diag().log().sum()
    chol = ax.axhline(correct, color="red", linestyle=":")

    handles = [chol]
    labels = ["log det Cholesky"]

    for label, model in models.items():
        lines = _plot_layer_ldj(model, ax, batch_size=batch_size)
        handles.append(lines)
        labels.append(label)

    ax.legend(handles=handles, labels=labels)

    fig.tight_layout()

    return fig
