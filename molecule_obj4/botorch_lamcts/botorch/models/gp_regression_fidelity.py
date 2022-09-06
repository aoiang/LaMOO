#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Gaussian Process Regression models based on GPyTorch models.

.. [Wu2019mf]
    J. Wu, S. Toscano-Palmerin, P. I. Frazier, and A. G. Wilson. Practical
    multi-fidelity bayesian optimization for hyperparameter tuning. ArXiv 2019.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.kernels.downsampling import DownsamplingKernel
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel
from botorch.models.kernels.linear_truncated_fidelity import (
    LinearTruncatedFidelityKernel,
)
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.containers import TrainingData
from gpytorch.kernels.kernel import ProductKernel
from gpytorch.kernels.rbf_kernel import RBFKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.priors.torch_priors import GammaPrior
from torch import Tensor


class SingleTaskMultiFidelityGP(SingleTaskGP):
    r"""A single task multi-fidelity GP model.

    A SingleTaskGP model using a DownsamplingKernel for the data fidelity
    parameter (if present) and an ExponentialDecayKernel for the iteration
    fidelity parameter (if present).

    This kernel is described in [Wu2019mf]_.

    Args:
        train_X: A `batch_shape x n x (d + s)` tensor of training features,
            where `s` is the dimension of the fidelity parameters (either one
            or two).
        train_Y: A `batch_shape x n x m` tensor of training observations.
        iteration_fidelity: The column index for the training iteration fidelity
            parameter (optional).
        data_fidelity: The column index for the downsampling fidelity parameter
            (optional).
        linear_truncated: If True, use a `LinearTruncatedFidelityKernel` instead
            of the default kernel.
        nu: The smoothness parameter for the Matern kernel: either 1/2, 3/2, or
            5/2. Only used when `linear_truncated=True`.
        likelihood: A likelihood. If omitted, use a standard GaussianLikelihood
            with inferred noise level.
        outcome_transform: An outcome transform that is applied to the
                training data during instantiation and to the posterior during
                inference (that is, the `Posterior` obtained by calling
                `.posterior` on the model will be on the original scale).
        input_transform: An input transform that is applied in the model's
                forward pass.

    Example:
        >>> train_X = torch.rand(20, 4)
        >>> train_Y = train_X.pow(2).sum(dim=-1, keepdim=True)
        >>> model = SingleTaskMultiFidelityGP(train_X, train_Y, data_fidelity=3)
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        iteration_fidelity: Optional[int] = None,
        data_fidelity: Optional[int] = None,
        linear_truncated: bool = True,
        nu: float = 2.5,
        likelihood: Optional[Likelihood] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        self._init_args = {
            "iteration_fidelity": iteration_fidelity,
            "data_fidelity": data_fidelity,
            "linear_truncated": linear_truncated,
            "nu": nu,
            "outcome_transform": outcome_transform,
        }
        if iteration_fidelity is None and data_fidelity is None:
            raise UnsupportedError(
                "SingleTaskMultiFidelityGP requires at least one fidelity parameter."
            )
        if input_transform is not None:
            input_transform.to(train_X)
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )

        self._set_dimensions(train_X=transformed_X, train_Y=train_Y)
        covar_module, subset_batch_dict = _setup_multifidelity_covar_module(
            dim=transformed_X.size(-1),
            aug_batch_shape=self._aug_batch_shape,
            iteration_fidelity=iteration_fidelity,
            data_fidelity=data_fidelity,
            linear_truncated=linear_truncated,
            nu=nu,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._subset_batch_dict = {
            "likelihood.noise_covar.raw_noise": -2,
            "mean_module.constant": -2,
            "covar_module.raw_outputscale": -1,
            **subset_batch_dict,
        }
        self.to(train_X)

    @classmethod
    def construct_inputs(cls, training_data: TrainingData, **kwargs) -> Dict[str, Any]:
        r"""Construct kwargs for the `Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: Options, expected for this class:
                - fidelity_features: List of columns of X that are fidelity parameters.
        """
        fidelity_features = kwargs.get("fidelity_features")
        if fidelity_features is None:
            raise ValueError(f"Fidelity features required for {cls.__name__}.")

        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "data_fidelity": fidelity_features[0],
        }


class FixedNoiseMultiFidelityGP(FixedNoiseGP):
    r"""A single task multi-fidelity GP model using fixed noise levels.

    A FixedNoiseGP model analogue to SingleTaskMultiFidelityGP, using a
    DownsamplingKernel for the data fidelity parameter (if present) and
    an ExponentialDecayKernel for the iteration fidelity parameter (if present).

    This kernel is described in [Wu2019mf]_.

    Args:
        train_X: A `batch_shape x n x (d + s)` tensor of training features,
            where `s` is the dimension of the fidelity parameters (either one
            or two).
        train_Y: A `batch_shape x n x m` tensor of training observations.
        train_Yvar: A `batch_shape x n x m` tensor of observed measurement noise.
        iteration_fidelity: The column index for the training iteration fidelity
            parameter (optional).
        data_fidelity: The column index for the downsampling fidelity parameter
            (optional).
        linear_truncated: If True, use a `LinearTruncatedFidelityKernel` instead
            of the default kernel.
        nu: The smoothness parameter for the Matern kernel: either 1/2, 3/2, or
            5/2. Only used when `linear_truncated=True`.
        outcome_transform: An outcome transform that is applied to the
            training data during instantiation and to the posterior during
            inference (that is, the `Posterior` obtained by calling
            `.posterior` on the model will be on the original scale).
        input_transform: An input transform that is applied in the model's
                forward pass.


    Example:
        >>> train_X = torch.rand(20, 4)
        >>> train_Y = train_X.pow(2).sum(dim=-1, keepdim=True)
        >>> train_Yvar = torch.full_like(train_Y) * 0.01
        >>> model = FixedNoiseMultiFidelityGP(
        >>>     train_X,
        >>>     train_Y,
        >>>     train_Yvar,
        >>>     data_fidelity=3,
        >>> )
    """

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Tensor,
        iteration_fidelity: Optional[int] = None,
        data_fidelity: Optional[int] = None,
        linear_truncated: bool = True,
        nu: float = 2.5,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        if iteration_fidelity is None and data_fidelity is None:
            raise UnsupportedError(
                "FixedNoiseMultiFidelityGP requires at least one fidelity parameter."
            )
        if input_transform is not None:
            input_transform.to(train_X)
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        self._set_dimensions(train_X=transformed_X, train_Y=train_Y)
        covar_module, subset_batch_dict = _setup_multifidelity_covar_module(
            dim=transformed_X.size(-1),
            aug_batch_shape=self._aug_batch_shape,
            iteration_fidelity=iteration_fidelity,
            data_fidelity=data_fidelity,
            linear_truncated=linear_truncated,
            nu=nu,
        )
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )
        self._subset_batch_dict = {
            "likelihood.noise_covar.raw_noise": -2,
            "mean_module.constant": -2,
            "covar_module.raw_outputscale": -1,
            **subset_batch_dict,
        }
        self.to(train_X)

    @classmethod
    def construct_inputs(cls, training_data: TrainingData, **kwargs) -> Dict[str, Any]:
        r"""Construct kwargs for the `Model` from `TrainingData` and other options.

        Args:
            training_data: `TrainingData` container with data for single outcome
                or for multiple outcomes for batched multi-output case.
            **kwargs: Options, expected for this class:
                - fidelity_features: List of columns of X that are fidelity parameters.
        """
        fidelity_features = kwargs.get("fidelity_features")
        if fidelity_features is None:
            raise ValueError(f"Fidelity features required for {cls.__name__}.")
        if training_data.Yvar is None:
            raise ValueError(f"Yvar required for {cls.__name__}.")

        return {
            "train_X": training_data.X,
            "train_Y": training_data.Y,
            "train_Yvar": training_data.Yvar,
            "data_fidelity": fidelity_features[0],
        }


def _setup_multifidelity_covar_module(
    dim: int,
    aug_batch_shape: torch.Size,
    iteration_fidelity: Optional[int],
    data_fidelity: Optional[int],
    linear_truncated: bool,
    nu: float,
) -> Tuple[ScaleKernel, Dict]:
    """Helper function to get the covariance module and associated subset_batch_dict
    for the multifidelity setting.

    Args:
        dim: The dimensionality of the training data.
        aug_batch_shape: The output-augmented batch shape as defined in
            `BatchedMultiOutputGPyTorchModel`.
        iteration_fidelity: The column index for the training iteration fidelity
            parameter (optional).
        data_fidelity: The column index for the downsampling fidelity parameter
            (optional).
        linear_truncated: If True, use a `LinearTruncatedFidelityKernel` instead
            of the default kernel.
        nu: The smoothness parameter for the Matern kernel: either 1/2, 3/2, or
            5/2. Only used when `linear_truncated=True`.

    Returns:
        The covariance module and subset_batch_dict.
    """

    if iteration_fidelity is not None and iteration_fidelity < 0:
        iteration_fidelity = dim + iteration_fidelity
    if data_fidelity is not None and data_fidelity < 0:
        data_fidelity = dim + data_fidelity

    if linear_truncated:
        fidelity_dims = [
            i for i in (iteration_fidelity, data_fidelity) if i is not None
        ]
        kernel = LinearTruncatedFidelityKernel(
            fidelity_dims=fidelity_dims,
            dimension=dim,
            nu=nu,
            batch_shape=aug_batch_shape,
            power_prior=GammaPrior(3.0, 3.0),
        )
    else:
        active_dimsX = [
            i for i in range(dim) if i not in {iteration_fidelity, data_fidelity}
        ]
        kernel = RBFKernel(
            ard_num_dims=len(active_dimsX),
            batch_shape=aug_batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
            active_dims=active_dimsX,
        )
        additional_kernels = []
        if iteration_fidelity is not None:
            exp_kernel = ExponentialDecayKernel(
                batch_shape=aug_batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
                offset_prior=GammaPrior(3.0, 6.0),
                power_prior=GammaPrior(3.0, 6.0),
                active_dims=[iteration_fidelity],
            )
            additional_kernels.append(exp_kernel)
        if data_fidelity is not None:
            ds_kernel = DownsamplingKernel(
                batch_shape=aug_batch_shape,
                offset_prior=GammaPrior(3.0, 6.0),
                power_prior=GammaPrior(3.0, 6.0),
                active_dims=[data_fidelity],
            )
            additional_kernels.append(ds_kernel)
        kernel = ProductKernel(kernel, *additional_kernels)

    covar_module = ScaleKernel(
        kernel, batch_shape=aug_batch_shape, outputscale_prior=GammaPrior(2.0, 0.15)
    )

    if linear_truncated:
        subset_batch_dict = {
            "covar_module.base_kernel.raw_power": -2,
            "covar_module.base_kernel.covar_module_unbiased.raw_lengthscale": -3,
            "covar_module.base_kernel.covar_module_biased.raw_lengthscale": -3,
        }
    else:
        subset_batch_dict = {
            "covar_module.base_kernel.kernels.0.raw_lengthscale": -3,
            "covar_module.base_kernel.kernels.1.raw_power": -2,
            "covar_module.base_kernel.kernels.1.raw_offset": -2,
        }
        if iteration_fidelity is not None:
            subset_batch_dict = {
                "covar_module.base_kernel.kernels.1.raw_lengthscale": -3,
                **subset_batch_dict,
            }
            if data_fidelity is not None:
                subset_batch_dict = {
                    "covar_module.base_kernel.kernels.2.raw_power": -2,
                    "covar_module.base_kernel.kernels.2.raw_offset": -2,
                    **subset_batch_dict,
                }

    return covar_module, subset_batch_dict
