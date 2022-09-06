#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional

import torch
from botorch.acquisition.objective import AcquisitionObjective
from botorch.exceptions.errors import BotorchError, BotorchTensorDimensionError
from botorch.models.transforms.outcome import Standardize
from botorch.posteriors import GPyTorchPosterior
from botorch.utils.transforms import normalize_indices
from torch import Tensor


class MCMultiOutputObjective(AcquisitionObjective):
    r"""Abstract base class for MC multi-output objectives."""

    @abstractmethod
    def forward(self, samples: Tensor, **kwargs) -> Tensor:
        r"""Evaluate the multi-output objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim Tensors of samples from
                a model posterior.

        Returns:
            A `sample_shape x batch_shape x q x m'`-dim Tensor of objective values with
            `m'` the output dimension. This assumes maximization in each output
            dimension).

        This method is usually not called directly, but via the objectives

        Example:
            >>> # `__call__` method:
            >>> samples = sampler(posterior)
            >>> outcomes = multi_obj(samples)
        """
        pass  # pragma: no cover


class IdentityMCMultiOutputObjective(MCMultiOutputObjective):
    r"""Trivial objective that returns the unaltered samples.

    Example:
        >>> identity_objective = IdentityMCMultiOutputObjective()
        >>> samples = sampler(posterior)
        >>> objective = identity_objective(samples)
    """

    def __init__(
        self, outcomes: Optional[List[int]] = None, num_outcomes: Optional[int] = None
    ) -> None:
        r"""Initialize Objective.

        Args:
            weights: `m'`-dim tensor of outcome weights.
            outcomes: A list of the `m'` indices that the weights should be
                applied to.
            num_outcomes: The total number of outcomes `m`
        """
        super().__init__()
        if outcomes is not None:
            if len(outcomes) < 2:
                raise BotorchTensorDimensionError(
                    "Must specify at least two outcomes for MOO."
                )
            if any(i < 0 for i in outcomes):
                if num_outcomes is None:
                    raise BotorchError(
                        "num_outcomes is required if any outcomes are less than 0."
                    )
                outcomes = normalize_indices(outcomes, num_outcomes)
            self.register_buffer("outcomes", torch.tensor(outcomes, dtype=torch.long))

    def forward(self, samples: Tensor) -> Tensor:
        if hasattr(self, "outcomes"):
            return samples.index_select(-1, self.outcomes.to(device=samples.device))
        return samples


class WeightedMCMultiOutputObjective(IdentityMCMultiOutputObjective):
    r"""Objective that reweights samples by given weights vector.

    Example:
        >>> weights = torch.tensor([1.0, -1.0])
        >>> weighted_objective = WeightedMCMultiOutputObjective(weights)
        >>> samples = sampler(posterior)
        >>> objective = weighted_objective(samples)
    """

    def __init__(
        self,
        weights: Tensor,
        outcomes: Optional[List[int]] = None,
        num_outcomes: Optional[int] = None,
    ) -> None:
        r"""Initialize Objective.

        Args:
            weights: `m'`-dim tensor of outcome weights.
            outcomes: A list of the `m'` indices that the weights should be
                applied to.
            num_outcomes: the total number of outcomes `m`
        """
        super().__init__(outcomes=outcomes, num_outcomes=num_outcomes)
        if weights.ndim != 1:
            raise BotorchTensorDimensionError(
                f"weights must be an 1-D tensor, but got {weights.shape}."
            )
        elif outcomes is not None and weights.shape[0] != len(outcomes):
            raise BotorchTensorDimensionError(
                "weights must contain the name number of elements as outcomes, "
                f"but got {weights.numel()} weights and {len(outcomes)} outcomes."
            )
        self.register_buffer("weights", weights)

    def forward(self, samples: Tensor) -> Tensor:
        samples = super().forward(samples=samples)
        return samples * self.weights.to(samples)


class UnstandardizeMCMultiOutputObjective(IdentityMCMultiOutputObjective):
    r"""Objective that unstandardizes the samples.

    TODO: remove this when MultiTask models support outcome transforms.

    Example:
        >>> unstd_objective = UnstandardizeMCMultiOutputObjective(Y_mean, Y_std)
        >>> samples = sampler(posterior)
        >>> objective = unstd_objective(samples)
    """

    def __init__(
        self, Y_mean: Tensor, Y_std: Tensor, outcomes: Optional[List[int]] = None
    ) -> None:
        r"""Initialize objective.

        Args:
            Y_mean: `m`-dim tensor of outcome means.
            Y_std: `m`-dim tensor of outcome standard deviations.
            outcomes: A list of `m' <= m` indices that specifies which of the `m` model
                outputs should be considered as the outcomes for MOO. If omitted, use
                all model outcomes. Typically used for constrained optimization.
        """
        if Y_mean.ndim > 1 or Y_std.ndim > 1:
            raise BotorchTensorDimensionError(
                "Y_mean and Y_std must both be 1-dimensional, but got "
                f"{Y_mean.ndim} and {Y_std.ndim}"
            )
        elif outcomes is not None and len(outcomes) > Y_mean.shape[-1]:
            raise BotorchTensorDimensionError(
                f"Cannot specify more ({len(outcomes)}) outcomes than are present in "
                f"the normalization inputs ({Y_mean.shape[-1]})."
            )
        super().__init__(outcomes=outcomes, num_outcomes=Y_mean.shape[-1])
        if outcomes is not None:
            Y_mean = Y_mean.index_select(-1, self.outcomes.to(Y_mean.device))
            Y_std = Y_std.index_select(-1, self.outcomes.to(Y_mean.device))

        self.register_buffer("Y_mean", Y_mean)
        self.register_buffer("Y_std", Y_std)

    def forward(self, samples: Tensor) -> Tensor:
        samples = super().forward(samples=samples)
        return samples * self.Y_std + self.Y_mean


class AnalyticMultiOutputObjective(AcquisitionObjective):
    r"""Abstract base class for multi-output analyic objectives."""

    @abstractmethod
    def forward(self, posterior: GPyTorchPosterior) -> GPyTorchPosterior:
        r"""Transform the posterior

        Args:
            posterior: A posterior.

        Returns:
            A transformed posterior.
        """
        pass  # pragma: no cover


class IdentityAnalyticMultiOutputObjective(AnalyticMultiOutputObjective):
    def forward(self, posterior: GPyTorchPosterior) -> GPyTorchPosterior:
        return posterior


class UnstandardizeAnalyticMultiOutputObjective(AnalyticMultiOutputObjective):
    r"""Objective that unstandardizes the posterior.

    TODO: remove this when MultiTask models support outcome transforms.

    Example:
        >>> unstd_objective = UnstandardizeAnalyticMultiOutputObjective(Y_mean, Y_std)
        >>> unstd_posterior = unstd_objective(posterior)
    """

    def __init__(self, Y_mean: Tensor, Y_std: Tensor) -> None:
        r"""Initialize objective.

        Args:
            Y_mean: `m`-dim tensor of outcome means
            Y_std: `m`-dim tensor of outcome standard deviations

        """
        if Y_mean.ndim > 1 or Y_std.ndim > 1:
            raise BotorchTensorDimensionError(
                "Y_mean and Y_std must both be 1-dimensional, but got "
                f"{Y_mean.ndim} and {Y_std.ndim}"
            )
        super().__init__()
        self.outcome_transform = Standardize(m=Y_mean.shape[0]).to(Y_mean)
        Y_std_unsqueezed = Y_std.unsqueeze(0)
        self.outcome_transform.means = Y_mean.unsqueeze(0)
        self.outcome_transform.stdvs = Y_std_unsqueezed
        self.outcome_transform._stdvs_sq = Y_std_unsqueezed.pow(2)
        self.outcome_transform.eval()

    def forward(self, posterior: GPyTorchPosterior) -> Tensor:
        return self.outcome_transform.untransform_posterior(posterior)
