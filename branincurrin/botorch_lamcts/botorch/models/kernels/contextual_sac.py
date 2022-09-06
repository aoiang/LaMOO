#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List

import torch
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.lazy.sum_lazy_tensor import SumLazyTensor
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor
from torch.nn import ModuleDict  # pyre-ignore


class SACKernel(Kernel):
    r"""The structural additive contextual(SAC) kernel.

    The kernel is used for contextual BO without oberseving context breakdowns.
    There are d parameters and M contexts. In total, the dimension of parameter space
    is d*M and input x can be written as
    x=[x_11, ..., x_1d, x_21, ..., x_2d, ...,  x_M1, ..., x_Md].

    The kernel uses the parameter decomposition and assumes an additive structure
    across contexts. Each context compponent is assumed to be independent.

    .. math::
       \begin{equation*}
          k(\mathbf{x}, \mathbf{x'}) = k_1(\mathbf{x_(1)}, \mathbf{x'_(1)}) + \cdots
          + k_M(\mathbf{x_(M)}, \mathbf{x'_(M)})
       \end{equation*}

    where
    * :math: M is the number of partitions of parameter space. Each partition contains
    same number of parameters d. Each kernel `k_i` acts only on d parameters of ith
    partition i.e. `\mathbf{x}_(i)`. Each kernel `k_i` is a scaled Matern kernel
    with same lengthscales but different outputscales.

    Args:
        decomposition: Keys are context names. Values are the indexes of parameters
            belong to the context. The parameter indexes are in the same order across
            contexts.
        batch_shape: Batch shape as usual for gpytorch kernels.
    """

    def __init__(
        self, decomposition: Dict[str, List[int]], batch_shape: torch.Size
    ) -> None:
        super().__init__(batch_shape=batch_shape)
        self.decomposition = decomposition

        num_param = len(next(iter(decomposition.values())))
        for active_parameters in decomposition.values():
            # check number of parameters are same in each decomp
            if len(active_parameters) != num_param:
                raise ValueError(
                    "num of parameters needs to be same across all contexts"
                )

        self._indexers = {
            context: torch.tensor(active_params)
            for context, active_params in self.decomposition.items()
        }

        self.base_kernel = MaternKernel(
            nu=2.5,
            ard_num_dims=num_param,
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
        )

        self.kernel_dict = {}  # scaled kernel for each parameter space partition
        for context in list(decomposition.keys()):
            self.kernel_dict[context] = ScaleKernel(
                base_kernel=self.base_kernel, outputscale_prior=GammaPrior(2.0, 15.0)
            )
        self.kernel_dict = ModuleDict(self.kernel_dict)

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params: Any,
    ) -> Tensor:
        """
        iterate across each partition of parameter space and sum the
        covariance matrices together
        """
        # same lengthscale for all the components
        covars = [
            self.kernel_dict[context](
                x1=x1.index_select(dim=-1, index=active_params),  # pyre-ignore
                x2=x2.index_select(dim=-1, index=active_params),
                diag=diag,
            )
            for context, active_params in self._indexers.items()
        ]

        if diag:
            res = sum(covars)
        else:
            res = SumLazyTensor(*covars)
        return res
