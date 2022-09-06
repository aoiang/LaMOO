#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Acquisition functions for max-value entropy search (MES) and
multi-fidelity MES with noisy observation and trace observations.

References

.. [Wang2018mves]
    Wang, Z., Jegelka, S.,
    Max-value Entropy Search for Efficient Bayesian Optimization.
    arXiv:1703.01968v3, 2018

.. [Takeno2019mfmves]
    Takeno, S., et al.,
    Multi-fidelity Bayesian Optimization with Max-value Entropy Search.
    arXiv:1901.08275v1, 2019
"""

from __future__ import annotations

from copy import deepcopy
from math import log
from typing import Any, Callable, Optional

import numpy as np
import torch
from botorch.acquisition.cost_aware import CostAwareUtility, InverseCostWeightedUtility
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.model import Model
from botorch.models.utils import check_no_nans
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.utils.transforms import match_batch_shape, t_batch_mode_transform
from gpytorch.utils.cholesky import psd_safe_cholesky
from scipy.optimize import brentq
from scipy.stats import norm
from torch import Tensor


CLAMP_LB = 1.0e-8


class qMaxValueEntropy(MCAcquisitionFunction):
    r"""The acquisition function for Max-value Entropy Search.

    This acquisition function computes the mutual information of
    max values and a candidate point X. See [Wang2018mves]_ for
    a detailed discussion.

    The model must be single-outcome.
    q > 1 is supported through cyclic optimization and fantasies.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        >>> MES = qMaxValueEntropy(model, candidate_set)
        >>> mes = MES(test_X)
    """

    def __init__(
        self,
        model: Model,
        candidate_set: Tensor,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        use_gumbel: bool = True,
        maximize: bool = True,
        X_pending: Optional[Tensor] = None,
        train_inputs: Tensor = None,
        **kwargs: Any,
    ) -> None:
        r"""Single-outcome max-value entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            candidate_set: A `n x d` Tensor including `n` candidate points to
                discretize the design space. Max values are sampled from the
                (joint) model posterior over these points.
            num_fantasies: Number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity, wall time and memory). Ignored if `X_pending` is `None`.
            num_mv_samples: Number of max value samples.
            num_y_samples: Number of posterior samples at specific design point `X`.
            use_gumbel: If True, use Gumbel approximation to sample the max values.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            maximize: If True, consider the problem a maximization problem.
            train_inputs: A `n_train x d` Tensor that the model has been fitted on,
                optional if model is an exact GP model.
        """
        sampler = SobolQMCNormalSampler(num_y_samples)
        super().__init__(model=model, sampler=sampler)

        # Batch GP models (e.g. fantasized models) are not currently supported
        if train_inputs is None:
            train_inputs = self.model.train_inputs[0]
        if train_inputs.ndim > 2:
            raise NotImplementedError(
                "Batch GP models (e.g. fantasized models) "
                "are not yet supported by qMaxValueEntropy"
            )

        self._init_model = model  # only used for the `fantasize()` in `set_X_pending()`
        train_inputs = match_batch_shape(train_inputs, candidate_set)
        self.candidate_set = torch.cat([candidate_set, train_inputs], dim=0)
        self.fantasies_sampler = SobolQMCNormalSampler(num_fantasies)
        self.num_fantasies = num_fantasies
        self.use_gumbel = use_gumbel
        self.num_mv_samples = num_mv_samples
        self.maximize = maximize
        self.weight = 1.0 if maximize else -1.0

        # If we put the `self._sample_max_values()` to `set_X_pending()`,
        # it will throw errors when the initial `super().__init__()` is called,
        # since some members required by `_sample_max_values()` are not yet initialized.
        if X_pending is None:
            self._sample_max_values()
        else:
            self.set_X_pending(X_pending)

    def set_X_pending(self, X_pending: Optional[Tensor] = None) -> None:
        r"""Set pending points.

        Informs the acquisition function about pending design points,
        fantasizes the model on the pending points and draws max-value samples
        from the fantasized model posterior.

        Args:
            X_pending: `m x d` Tensor with `m` `d`-dim design points that have
                been submitted for evaluation but have not yet been evaluated.
        """
        super().set_X_pending(X_pending=X_pending)
        if X_pending is not None:
            # fantasize the model
            fantasy_model = self._init_model.fantasize(
                X=X_pending, sampler=self.fantasies_sampler, observation_noise=True
            )
            self.model = fantasy_model
            self._sample_max_values()
        else:
            # This is mainly for setting the model to the original model
            # after the sequential optimization at q > 1
            try:
                self.model = self._init_model
                self._sample_max_values()
            except AttributeError:
                pass

    def _sample_max_values(self):
        r"""Sample max values for MC approximation of the expectation in MES"""
        with torch.no_grad():
            # Append X_pending to candidate set
            if self.X_pending is None:
                X_pending = torch.tensor(
                    [], dtype=self.candidate_set.dtype, device=self.candidate_set.device
                )
            else:
                X_pending = self.X_pending
            X_pending = match_batch_shape(X_pending, self.candidate_set)
            candidate_set = torch.cat([self.candidate_set, X_pending], dim=0)

            # project the candidate_set to the highest fidelity,
            # which is needed for the multi-fidelity MES
            try:
                candidate_set = self.project(candidate_set)
            except AttributeError:
                pass

            # sample max values
            if self.use_gumbel:
                self.posterior_max_values = _sample_max_value_Gumbel(
                    self.model, candidate_set, self.num_mv_samples, self.maximize
                )
            else:
                self.posterior_max_values = _sample_max_value_Thompson(
                    self.model, candidate_set, self.num_mv_samples, self.maximize
                )

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Compute max-value entropy at the design points `X`.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design points each.

        Returns:
            A `batch_shape`-dim Tensor of MVE values at the given design points `X`.
        """
        # Compute the posterior, posterior mean, variance and std
        posterior = self.model.posterior(X.unsqueeze(-3), observation_noise=False)
        mean = self.weight * posterior.mean.squeeze(-1).squeeze(-1)
        # batch_shape x num_fantasies
        variance = posterior.variance.clamp_min(CLAMP_LB).view_as(mean)
        check_no_nans(mean)
        check_no_nans(variance)

        ig = self._compute_information_gain(
            X=X, mean_M=mean, variance_M=variance, covar_mM=variance.unsqueeze(-1)
        )

        return ig.mean(dim=0)  # average over the fantasies

    def _compute_information_gain(
        self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor
    ) -> Tensor:
        r"""Computes the information gain at the design points `X`.

        Approximately computes the information gain at the design points `X`,
        for both MES with noisy observations and multi-fidelity MES with noisy
        observation and trace observations.

        The implementation is inspired from the paper on multi-fidelity MES by
        Takeno et. al. [Takeno2019mfmves]_. The notations in the comments in this
        function follows the Appendix A in the paper.

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design point each.
            mean_M, variance_M: `batch_shape x num_fantasies`-dim Tensors of
                `batch_shape` t-batches with `num_fantasies` fantasies.
                `num_fantasies = 1` for non-fantasized models.
                All are obtained without noise.
            covar_mM: `batch_shape x num_fantasies x (1 + num_trace_observations)`
                -dim Tensor. `num_fantasies = 1` for non-fantasized models.
                All are obtained without noise.

        Returns:
            A `num_fantasies x batch_shape`-dim Tensor of information gains at the
            given design points `X`.
        """

        # compute the std_m, variance_m with noisy observation
        posterior_m = self.model.posterior(X.unsqueeze(-3), observation_noise=True)
        mean_m = self.weight * posterior_m.mean.squeeze(-1)
        # batch_shape x num_fantasies x (1 + num_trace_observations)
        variance_m = posterior_m.mvn.covariance_matrix
        # batch_shape x num_fantasies x (1 + num_trace_observations)^2
        check_no_nans(variance_m)

        # compute mean and std for fM|ym, x, Dt ~ N(u, s^2)
        samples_m = self.weight * self.sampler(posterior_m).squeeze(-1)
        # s_m x batch_shape x num_fantasies x (1 + num_trace_observations)
        L = psd_safe_cholesky(variance_m)
        temp_term = torch.cholesky_solve(covar_mM.unsqueeze(-1), L).transpose(-2, -1)
        # equivalent to torch.matmul(covar_mM.unsqueeze(-2), torch.inverse(variance_m))
        # batch_shape x num_fantasies x 1 x (1 + num_trace_observations)

        mean_pt1 = torch.matmul(temp_term, (samples_m - mean_m).unsqueeze(-1))
        mean_new = mean_pt1.squeeze(-1).squeeze(-1) + mean_M
        # s_m x batch_shape x num_fantasies
        variance_pt1 = torch.matmul(temp_term, covar_mM.unsqueeze(-1))
        variance_new = variance_M - variance_pt1.squeeze(-1).squeeze(-1)
        # batch_shape x num_fantasies
        stdv_new = variance_new.clamp_min(CLAMP_LB).sqrt()
        # batch_shape x num_fantasies

        # define normal distribution to compute cdf and pdf
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # Compute p(fM <= f* | ym, x, Dt)
        view_shape = (
            [self.num_mv_samples] + [1] * (len(X.shape) - 2) + [self.num_fantasies]
        )  # s_M x batch_shape x num_fantasies
        if self.X_pending is None:
            view_shape[-1] = 1
        max_vals = self.posterior_max_values.view(view_shape).unsqueeze(1)
        # s_M x 1 x batch_shape x num_fantasies
        normalized_mvs_new = (max_vals - mean_new) / stdv_new
        # s_M x s_m x batch_shape x num_fantasies =
        # s_M x 1 x batch_shape x num_fantasies - s_m x batch_shape x num_fantasies
        cdf_mvs_new = normal.cdf(normalized_mvs_new).clamp_min(CLAMP_LB)

        # Compute p(fM <= f* | x, Dt)
        stdv_M = variance_M.sqrt()
        normalized_mvs = (max_vals - mean_M) / stdv_M
        # s_M x 1 x batch_shape x num_fantasies =
        # s_M x 1 x 1 x num_fantasies - batch_shape x num_fantasies
        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        # s_M x 1 x batch_shape x num_fantasies

        # Compute log(p(ym | x, Dt))
        log_pdf_fm = posterior_m.mvn.log_prob(self.weight * samples_m).unsqueeze(0)
        # 1 x s_m x batch_shape x num_fantasies

        # H0 = H(ym | x, Dt)
        H0 = posterior_m.mvn.entropy()  # batch_shape x num_fantasies

        # regression adjusted H1 estimation, H1_hat = H1_bar - beta * (H0_bar - H0)
        # H1 = E_{f*|x, Dt}[H(ym|f*, x, Dt)]
        Z = cdf_mvs_new / cdf_mvs  # s_M x s_m x batch_shape x num_fantasies
        h1 = -Z * Z.log() - Z * log_pdf_fm  # s_M x s_m x batch_shape x num_fantasies
        check_no_nans(h1)
        dim = [0, 1]  # dimension of fm samples, fM samples
        H1_bar = h1.mean(dim=dim)
        h0 = -log_pdf_fm
        H0_bar = h0.mean(dim=dim)
        cov = ((h1 - H1_bar) * (h0 - H0_bar)).mean(dim=dim)
        beta = cov / (h0.var(dim=dim) * h1.var(dim=dim)).sqrt()
        H1_hat = H1_bar - beta * (H0_bar - H0)
        ig = H0 - H1_hat  # batch_shape x num_fantasies
        ig = ig.permute(-1, *range(ig.dim() - 1))  # num_fantasies x batch_shape
        return ig


class qMultiFidelityMaxValueEntropy(qMaxValueEntropy):
    r"""Multi-fidelity max-value entropy.

    The acquisition function for multi-fidelity max-value entropy search
    with support for trace observations. See [Takeno2019mfmves]_ for a
    detailed discussion of the basic ideas on multi-fidelity MES
    (note that this implementation is somewhat different).

    The model must be single-outcome.
    q > 1 is supported through cyclic optimization and fantasies.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> candidate_set = torch.rand(1000, bounds.size(1))
        >>> candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        >>> MF_MES = qMultiFidelityMaxValueEntropy(model, candidate_set)
        >>> mf_mes = MF_MES(test_X)
    """

    def __init__(
        self,
        model: Model,
        candidate_set: Tensor,
        num_fantasies: int = 16,
        num_mv_samples: int = 10,
        num_y_samples: int = 128,
        use_gumbel: bool = True,
        X_pending: Optional[Tensor] = None,
        maximize: bool = True,
        cost_aware_utility: Optional[CostAwareUtility] = None,
        project: Callable[[Tensor], Tensor] = lambda X: X,
        expand: Callable[[Tensor], Tensor] = lambda X: X,
        **kwargs: Any,
    ) -> None:
        r"""Single-outcome max-value entropy search acquisition function.

        Args:
            model: A fitted single-outcome model.
            candidate_set: A `n x d` Tensor including `n` candidate points to
                discretize the design space, which will be used to sample the
                max values from their posteriors.
            cost_aware_utility: A CostAwareUtility computing the cost-transformed
                utility from a candidate set and samples of increases in utility.
            num_fantasies: Number of fantasies to generate. The higher this
                number the more accurate the model (at the expense of model
                complexity and performance) and it's only used when `X_pending`
                is not `None`.
            num_mv_samples: Number of max value samples.
            num_y_samples: Number of posterior samples at specific design point `X`.
            use_gumbel: If True, use Gumbel approximation to sample the max values.
            X_pending: A `m x d`-dim Tensor of `m` design points that have been
                submitted for function evaluation but have not yet been evaluated.
            maximize: If True, consider the problem a maximization problem.
            cost_aware_utility: A CostAwareUtility computing the cost-transformed
                utility from a candidate set and samples of increases in utility.
            project: A callable mapping a `batch_shape x q x d` tensor of design
                points to a tensor of the same shape projected to the desired
                target set (e.g. the target fidelities in case of multi-fidelity
                optimization).
            expand: A callable mapping a `batch_shape x q x d` input tensor to
                a `batch_shape x (q + q_e)' x d`-dim output tensor, where the
                `q_e` additional points in each q-batch correspond to
                additional ("trace") observations.
        """
        super().__init__(
            model=model,
            candidate_set=candidate_set,
            num_fantasies=num_fantasies,
            num_mv_samples=num_mv_samples,
            num_y_samples=num_y_samples,
            X_pending=X_pending,
            use_gumbel=use_gumbel,
            maximize=maximize,
        )

        if cost_aware_utility is None:
            cost_model = AffineFidelityCostModel(fidelity_weights={-1: 1.0})
            cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

        self.cost_aware_utility = cost_aware_utility
        self.expand = expand
        self.project = project
        self._cost_sampler = None

        # @TODO make sure fidelity_dims align in project, expand & cost_aware_utility
        # It seems very difficult due to the current way of handling project/expand

        # resample max values after initializing self.project
        # so that the max value samples are at the highest fidelity
        self._sample_max_values()

    @property
    def cost_sampler(self):
        if self._cost_sampler is None:
            # Note: Using the deepcopy here is essential. Removing this poses a
            # problem if the base model and the cost model have a different number
            # of outputs or test points (this would be caused by expand), as this
            # would trigger re-sampling the base samples in the fantasy sampler.
            # By cloning the sampler here, the right thing will happen if the
            # the sizes are compatible, if they are not this will result in
            # samples being drawn using different base samples, but it will at
            # least avoid changing state of the fantasy sampler.
            self._cost_sampler = deepcopy(self.fantasies_sampler)
        return self._cost_sampler

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluates `qMultifidelityMaxValueEntropy` at the design points `X`

        Args:
            X: A `batch_shape x 1 x d`-dim Tensor of `batch_shape` t-batches
                with `1` `d`-dim design point each.

        Returns:
            A `batch_shape`-dim Tensor of MF-MVES values at the design points `X`.
        """
        X_expand = self.expand(X)  # batch_shape x (1 + num_trace_observations) x d
        X_max_fidelity = self.project(X)  # batch_shape x 1 x d
        X_all = torch.cat((X_expand, X_max_fidelity), dim=-2).unsqueeze(-3)
        # batch_shape x num_fantasies x (2 + num_trace_observations) x d

        # Compute the posterior, posterior mean, variance without noise
        # `_m` and `_M` in the var names means the current and the max fidelity.
        posterior = self.model.posterior(X_all, observation_noise=False)
        mean_M = self.weight * posterior.mean[..., -1, 0]  # batch_shape x num_fantasies
        variance_M = posterior.variance[..., -1, 0].clamp_min(CLAMP_LB)
        # get the covariance between the low fidelities and max fidelity
        covar_mM = posterior.mvn.covariance_matrix[..., :-1, -1]
        # batch_shape x num_fantasies x (1 + num_trace_observations)

        check_no_nans(mean_M)
        check_no_nans(variance_M)
        check_no_nans(covar_mM)

        # compute the information gain (IG)
        ig = self._compute_information_gain(
            X=X_expand, mean_M=mean_M, variance_M=variance_M, covar_mM=covar_mM
        )
        ig = self.cost_aware_utility(X=X, deltas=ig, sampler=self.cost_sampler)
        return ig.mean(dim=0)  # average over the fantasies


def _sample_max_value_Thompson(
    model: Model, candidate_set: Tensor, num_samples: int, maximize: bool = True
) -> Tensor:
    """Samples the max values by discrete Thompson sampling.

    Should generally be called within a `with torch.no_grad()` context.

    Args:
        model: A fitted single-outcome model.
        candidate_set: A `n x d` Tensor including `n` candidate points to
            discretize the design space.
        num_samples: Number of max value samples.
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A `num_samples x num_fantasies` Tensor of max value samples
    """
    posterior = model.posterior(candidate_set)
    weight = 1.0 if maximize else -1.0
    samples = weight * posterior.rsample(torch.Size([num_samples])).squeeze(-1)
    # samples is num_samples x (num_fantasies) x n
    max_values, _ = samples.max(dim=-1)
    if len(samples.shape) == 2:
        max_values = max_values.unsqueeze(-1)  # num_samples x num_fantasies

    return max_values


def _sample_max_value_Gumbel(
    model: Model, candidate_set: Tensor, num_samples: int, maximize: bool = True
) -> Tensor:
    """Samples the max values by Gumbel approximation.

    Should generally be called within a `with torch.no_grad()` context.

    Args:
        model: A fitted single-outcome model.
        candidate_set: A `n x d` Tensor including `n` candidate points to
            discretize the design space.
        num_samples: Number of max value samples.
        maximize: If True, consider the problem a maximization problem.

    Returns:
        A `num_samples x num_fantasies` Tensor of max value samples
    """
    # define the approximate CDF for the max value under the independence assumption
    posterior = model.posterior(candidate_set)
    weight = 1.0 if maximize else -1.0
    mu = weight * posterior.mean
    sigma = posterior.variance.clamp_min(1e-8).sqrt()
    # mu, sigma is (num_fantasies) X n X 1
    if len(mu.shape) == 3 and mu.shape[-1] == 1:
        mu = mu.squeeze(-1).T
        sigma = sigma.squeeze(-1).T
    # mu, sigma is now n X num_fantasies or n X 1

    # bisect search to find the quantiles 25, 50, 75
    lo = (mu - 3 * sigma).min(dim=0).values
    hi = (mu + 5 * sigma).max(dim=0).values
    num_fantasies = mu.shape[1]
    device = candidate_set.device
    dtype = candidate_set.dtype
    quantiles = torch.zeros(num_fantasies, 3, device=device, dtype=dtype)
    for i in range(num_fantasies):
        lo_, hi_ = lo[i], hi[i]
        N = norm(mu[:, i].cpu().numpy(), sigma[:, i].cpu().numpy())
        quantiles[i, :] = torch.tensor(
            [
                brentq(lambda y: np.exp(np.sum(N.logcdf(y))) - p, lo_, hi_)
                for p in [0.25, 0.50, 0.75]
            ]
        )
    q25, q50, q75 = quantiles[:, 0], quantiles[:, 1], quantiles[:, 2]
    # q25, q50, q75 are 1 dimensional tensor with size of either 1 or num_fantasies

    # parameter fitting based on matching percentiles for the Gumbel distribution
    b = (q25 - q75) / (log(log(4.0 / 3.0)) - log(log(4.0)))
    a = q50 + b * log(log(2.0))

    # inverse sampling from the fitted Gumbel CDF distribution
    sample_shape = (num_samples, num_fantasies)
    eps = torch.rand(*sample_shape, device=device, dtype=dtype)
    max_values = a - b * eps.log().mul(-1.0).log()

    return max_values  # num_samples x num_fantasies
