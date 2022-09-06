#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings

import torch
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_model
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.utils.testing import BotorchTestCase
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


EPS = 1e-8

NOISE = [
    [0.127],
    [-0.113],
    [-0.345],
    [-0.034],
    [-0.069],
    [-0.272],
    [0.013],
    [0.056],
    [0.087],
    [-0.081],
]


class TestEndToEnd(BotorchTestCase):
    def _setUp(self, double=False):
        dtype = torch.double if double else torch.float
        train_x = torch.linspace(0, 1, 10, device=self.device, dtype=dtype).unsqueeze(
            -1
        )
        train_y = torch.sin(train_x * (2 * math.pi))
        train_yvar = torch.tensor(0.1 ** 2, device=self.device)
        noise = torch.tensor(NOISE, device=self.device, dtype=dtype)
        self.train_x = train_x
        self.train_y = train_y + noise
        self.train_yvar = train_yvar
        self.bounds = torch.tensor([[0.0], [1.0]], device=self.device, dtype=dtype)
        model_st = SingleTaskGP(self.train_x, self.train_y)
        self.model_st = model_st.to(device=self.device, dtype=dtype)
        self.mll_st = ExactMarginalLogLikelihood(
            self.model_st.likelihood, self.model_st
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            self.mll_st = fit_gpytorch_model(
                self.mll_st, options={"maxiter": 5}, max_retries=1
            )
        model_fn = FixedNoiseGP(
            self.train_x, self.train_y, self.train_yvar.expand_as(self.train_y)
        )
        self.model_fn = model_fn.to(device=self.device, dtype=dtype)
        self.mll_fn = ExactMarginalLogLikelihood(
            self.model_fn.likelihood, self.model_fn
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            self.mll_fn = fit_gpytorch_model(
                self.mll_fn, options={"maxiter": 5}, max_retries=1
            )

    def test_qEI(self):
        for double in (True, False):
            self._setUp(double=double)
            qEI = qExpectedImprovement(self.model_st, best_f=0.0)
            candidates, _ = optimize_acqf(
                acq_function=qEI,
                bounds=self.bounds,
                q=3,
                num_restarts=10,
                raw_samples=20,
                options={"maxiter": 5},
            )
            self.assertTrue(torch.all(-EPS <= candidates))
            self.assertTrue(torch.all(candidates <= 1 + EPS))
            qEI = qExpectedImprovement(self.model_fn, best_f=0.0)
            candidates, _ = optimize_acqf(
                acq_function=qEI,
                bounds=self.bounds,
                q=3,
                num_restarts=10,
                raw_samples=20,
                options={"maxiter": 5},
            )
            self.assertTrue(torch.all(-EPS <= candidates))
            self.assertTrue(torch.all(candidates <= 1 + EPS))
            candidates_batch_limit, _ = optimize_acqf(
                acq_function=qEI,
                bounds=self.bounds,
                q=3,
                num_restarts=10,
                raw_samples=20,
                options={"maxiter": 5, "batch_limit": 5},
            )
            self.assertTrue(torch.all(-EPS <= candidates_batch_limit))
            self.assertTrue(torch.all(candidates_batch_limit <= 1 + EPS))

    def test_EI(self):
        for double in (True, False):
            self._setUp(double=double)
            EI = ExpectedImprovement(self.model_st, best_f=0.0)
            candidates, _ = optimize_acqf(
                acq_function=EI,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=20,
                options={"maxiter": 5},
            )
            self.assertTrue(-EPS <= candidates <= 1 + EPS)
            EI = ExpectedImprovement(self.model_fn, best_f=0.0)
            candidates, _ = optimize_acqf(
                acq_function=EI,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=20,
                options={"maxiter": 5},
            )
            self.assertTrue(-EPS <= candidates <= 1 + EPS)
