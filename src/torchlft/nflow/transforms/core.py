from abc import ABCMeta, abstractmethod
from typing import Any, Callable, ClassVar, TypeAlias

import torch
import torch.nn as nn


Tensor: TypeAlias = torch.Tensor


class Transform:
    n_params: ClassVar[int]
    handle_params_fn: ClassVar[Callable[Tensor, Tensor]]
    transform_fn: ClassVar[
        Callable[[Tensor, Tensor, Any, ...], tuple[Tensor, Tensor]]
    ]

    def __init__(self, params: Tensor, **context):
        self.params = self.handle_params_fn(params)
        self.context = context

    def __call__(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        return self.transform_fn(inputs, self.params, **self.context)


# TODO: inverse?

Wrapper: TypeAlias = Callable[Transform, Transform]


def hook_from_wrapper(wrapper):
    def hook(module, inputs, outputs):
        return wrapper(outputs)

    return hook


class TransformModule(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, context: Any) -> Transform: ...


class UnivariateTransformModule(TransformModule):
    def __init__(
        self,
        transform_cls: type[Transform],
        context_fn: Callable[Tensor, Tensor],
        wrappers: list[Wrapper] = [],
    ):
        super().__init__()
        self.transform_cls = transform_cls
        self.context_fn = context_fn
        self._hooks = {
            wrapper.__name__: self.register_forward_hook(
                hook_from_wrapper(wrapper)
            )
            for wrapper in wrappers
        }

    def extra_repr(self) -> str:
        return f"(univariate): {self.transform_cls}"

    def forward(self, context: Any) -> Transform:
        params = self.context_fn(context)
        transform = self.transform_cls(params)
        return transform


# TODO: invertible 1x1 conv option to mix between channels (Real, Imag)
# TODO: this can trivially be generalised to autoregressive transform in
# arbitrary dimensions
def _autoregressive_bivariate_transform(
    univariate_a: UnivariateTransformModule,
    univariate_b: UnivariateTransformModule,
):
    class BivariateTransform:
        def __init__(self, context: Tensor):
            self.context = context

        def __call__(self, x: Tensor):
            assert x.shape[-1] == 2
            x_a, x_b = x.split(1, dim=-1)

            context = self.context
            y_a, ldj_a = univariate_a(context)(x_a)

            context = torch.cat([self.context, y_a], dim=-1)
            y_b, ldj_b = univariate_b(context)(x_b)

            y = torch.cat([y_a, y_b], dim=-1)
            ldj = ldj_a + ldj_b

            return y, ldj

    return BivariateTransform


class BivariateTransformModule(TransformModule):
    def __init__(
        self,
        univariate_a: UnivariateTransformModule,
        univariate_b: UnivariateTransformModule,
        context_fn: Callable[Tensor, Tensor] | None,
        wrappers: list[Wrapper] = [],
    ):
        super().__init__()
        self.register_module("univariate_a", univariate_a)
        self.register_module("univariate_b", univariate_b)

        self.context_fn = (
            context_fn if context_fn is not None else nn.Identity()
        )

        self.transform_factory = _autoregressive_bivariate_transform(
            self.univariate_a, self.univariate_b
        )

        for wrapper in wrappers:
            self.register_forward_hook(hook_from_wrapper(wrapper))
