#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
A wrapper around AquisitionFunctions to fix certain features for optimization.
This is useful e.g. for performing contextual optimization.
"""

from __future__ import annotations

from typing import List, Union

import torch
from botorch.acquisition.acquisition import AcquisitionFunction
from torch import Tensor
from torch.nn import Module


class FixedFeatureAcquisitionFunction(AcquisitionFunction):
    """A wrapper around AquisitionFunctions to fix a subset of features.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)  # d = 5
        >>> qEI = qExpectedImprovement(model, best_f=0.0)
        >>> columns = [2, 4]
        >>> values = X[..., columns]
        >>> qEI_FF = FixedFeatureAcquisitionFunction(qEI, 5, columns, values)
        >>> qei = qEI_FF(test_X)  # d' = 3
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        d: int,
        columns: List[int],
        values: Union[Tensor, List[float]],
    ) -> None:
        r"""Derived Acquisition Function by fixing a subset of input features.

        Args:
            acq_function: The base acquisition function, operating on input
                tensors `X_full` of feature dimension `d`.
            d: The feature dimension expected by `acq_function`.
            columns: `d_f < d` indices of columns in `X_full` that are to be
                fixed to the provided values.
            values: The values to which to fix the columns in `columns`. Either
                a full `batch_shape x q x d_f` tensor of values (if values are
                different for each of the `q` input points), or an array-like of
                values that is broadcastable to the input across `t`-batch and
                `q`-batch dimensions, e.g. a list of length `d_f` if values
                are the same across all `t` and `q`-batch dimensions.
        """
        Module.__init__(self)
        self.acq_func = acq_function
        self.d = d
        values = torch.as_tensor(values).detach().clone()
        self.register_buffer("values", values)
        # build selector for _construct_X_full
        self._selector = []
        idx_X, idx_f = 0, d - values.shape[-1]
        for i in range(self.d):
            if i in columns:
                self._selector.append(idx_f)
                idx_f += 1
            else:
                self._selector.append(idx_X)
                idx_X += 1

    def forward(self, X: Tensor):
        r"""Evaluate base acquisition function under the fixed features.

        Args:
            X: Input tensor of feature dimension `d' < d` such that `d' + d_f = d`.

        Returns:
            Base acquisition function evaluated on tensor `X_full` constructed
            by adding `values` in the appropriate places (see
            `_construct_X_full`).
        """
        X_full = self._construct_X_full(X)
        return self.acq_func(X_full)

    def _construct_X_full(self, X: Tensor) -> Tensor:
        r"""Constructs the full input for the base acquisition function.

        Args:
            X: Input tensor with shape `batch_shape x q x d'` such that
                `d' + d_f = d`.

        Returns:
            Tensor `X_full` of shape `batch_shape x q x d`, where
            `X_full[..., i] = values[..., i]` if `i in columns`,
            and `X_full[..., i] = X[..., j]`, with
            `j = i - sum_{l<=i} 1_{l in fixed_colunns}`.
        """
        d_prime, d_f = X.shape[-1], self.values.shape[-1]
        if d_prime + d_f != self.d:
            raise ValueError(
                f"Feature dimension d' ({d_prime}) of input must be "
                f"d - d_f ({self.d - d_f})."
            )
        # concatenate values to the end
        values = self.values.to(X).expand(*X.shape[:-1], d_f)
        X_perm = torch.cat([X, values], dim=-1)
        # now select the appropriate column order
        return X_perm[..., self._selector]
