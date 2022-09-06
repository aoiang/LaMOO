#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from botorch.test_functions.base import BaseTestProblem, ConstrainedBaseTestProblem
from botorch.utils.testing import BotorchTestCase
from torch import Tensor


class DummyTestProblem(BaseTestProblem):
    dim = 2
    _bounds = [(0, 1), (2, 3)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        return -X.pow(2).sum(dim=-1)


class DummyConstrainedTestProblem(DummyTestProblem, ConstrainedBaseTestProblem):

    num_constraints = 1

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        return 0.25 - X.sum(dim=-1, keepdim=True)


class TestBaseTestProblems(BotorchTestCase):
    def test_base_test_problem(self):
        for dtype in (torch.float, torch.double):
            problem = DummyTestProblem()
            self.assertIsNone(problem.noise_std)
            self.assertFalse(problem.negate)
            bnds_expected = torch.tensor([(0, 2), (1, 3)], dtype=torch.float)
            self.assertTrue(torch.equal(problem.bounds, bnds_expected))
            problem = problem.to(device=self.device, dtype=dtype)
            bnds_expected = bnds_expected.to(device=self.device, dtype=dtype)
            self.assertTrue(torch.equal(problem.bounds, bnds_expected))
            X = torch.rand(2, 2, device=self.device, dtype=dtype)
            Y = problem(X)
            self.assertTrue(torch.allclose(Y, -X.pow(2).sum(dim=-1)))
            problem = DummyTestProblem(negate=True, noise_std=0.1)
            self.assertEqual(problem.noise_std, 0.1)
            self.assertTrue(problem.negate)

    def test_constrained_base_test_problem(self):
        for dtype in (torch.float, torch.double):
            problem = DummyConstrainedTestProblem().to(device=self.device, dtype=dtype)
            X = torch.tensor([[0.4, 0.6], [0.1, 0.1]])
            feas = problem.is_feasible(X=X)
            self.assertFalse(feas[0].item())
            self.assertTrue(feas[1].item())
            problem = DummyConstrainedTestProblem(noise_std=0.0).to(
                device=self.device, dtype=dtype
            )
            feas = problem.is_feasible(X=X)
            self.assertFalse(feas[0].item())
            self.assertTrue(feas[1].item())
