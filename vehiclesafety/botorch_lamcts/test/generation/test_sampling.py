#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import itertools
from unittest import mock

import torch
from botorch.acquisition.analytic import PosteriorMean
from botorch.acquisition.objective import (
    IdentityMCObjective,
    LinearMCObjective,
    ScalarizedObjective,
)
from botorch.generation.sampling import (
    BoltzmannSampling,
    MaxPosteriorSampling,
    SamplingStrategy,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class TestSamplingStrategy(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            SamplingStrategy()


class TestMaxPosteriorSampling(BotorchTestCase):
    def test_init(self):
        mm = MockModel(MockPosterior(mean=None))
        MPS = MaxPosteriorSampling(mm)
        self.assertEqual(MPS.model, mm)
        self.assertTrue(MPS.replacement)
        self.assertIsInstance(MPS.objective, IdentityMCObjective)
        obj = LinearMCObjective(torch.rand(2))
        MPS = MaxPosteriorSampling(mm, objective=obj, replacement=False)
        self.assertEqual(MPS.objective, obj)
        self.assertFalse(MPS.replacement)

    def test_max_posterior_sampling(self):
        batch_shapes = (torch.Size(), torch.Size([3]), torch.Size([3, 2]))
        dtypes = (torch.float, torch.double)
        for batch_shape, dtype, N, num_samples, d in itertools.product(
            batch_shapes, dtypes, (5, 6), (1, 2), (1, 2)
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            # X is `batch_shape x N x d` = batch_shape x N x 1.
            X = torch.randn(*batch_shape, N, d, **tkwargs)
            # the event shape is `num_samples x batch_shape x N x m`
            psamples = torch.zeros(num_samples, *batch_shape, N, 1, **tkwargs)
            psamples[..., 0, :] = 1.0

            # IdentityMCObjective, with replacement
            with mock.patch.object(MockPosterior, "rsample", return_value=psamples):
                mp = MockPosterior(None)
                with mock.patch.object(MockModel, "posterior", return_value=mp):
                    mm = MockModel(None)
                    MPS = MaxPosteriorSampling(mm)
                    s = MPS(X, num_samples=num_samples)
                    self.assertTrue(torch.equal(s, X[..., [0] * num_samples, :]))

            # ScalarizedMCObjective, with replacement
            with mock.patch.object(MockPosterior, "rsample", return_value=psamples):
                mp = MockPosterior(None)
                with mock.patch.object(MockModel, "posterior", return_value=mp):
                    mm = MockModel(None)
                    with mock.patch.object(
                        ScalarizedObjective, "forward", return_value=mp
                    ):
                        obj = ScalarizedObjective(torch.rand(2, **tkwargs))
                        MPS = MaxPosteriorSampling(mm, objective=obj)
                        s = MPS(X, num_samples=num_samples)
                        self.assertTrue(torch.equal(s, X[..., [0] * num_samples, :]))

            # without replacement
            psamples[..., 1, 0] = 1e-6
            with mock.patch.object(MockPosterior, "rsample", return_value=psamples):
                mp = MockPosterior(None)
                with mock.patch.object(MockModel, "posterior", return_value=mp):
                    mm = MockModel(None)
                    MPS = MaxPosteriorSampling(mm, replacement=False)
                    if len(batch_shape) > 1:
                        with self.assertRaises(NotImplementedError):
                            MPS(X, num_samples=num_samples)
                    else:
                        s = MPS(X, num_samples=num_samples)
                        # order is not guaranteed, need to sort
                        self.assertTrue(
                            torch.equal(
                                torch.sort(s, dim=-2).values,
                                torch.sort(X[..., :num_samples, :], dim=-2).values,
                            )
                        )

            # ScalarizedMCObjective, without replacement
            with mock.patch.object(MockPosterior, "rsample", return_value=psamples):
                mp = MockPosterior(None)
                with mock.patch.object(MockModel, "posterior", return_value=mp):
                    mm = MockModel(None)
                    with mock.patch.object(
                        ScalarizedObjective, "forward", return_value=mp
                    ):
                        obj = ScalarizedObjective(torch.rand(2, **tkwargs))
                        MPS = MaxPosteriorSampling(mm, objective=obj, replacement=False)
                        if len(batch_shape) > 1:
                            with self.assertRaises(NotImplementedError):
                                MPS(X, num_samples=num_samples)
                        else:
                            s = MPS(X, num_samples=num_samples)
                            # order is not guaranteed, need to sort
                            self.assertTrue(
                                torch.equal(
                                    torch.sort(s, dim=-2).values,
                                    torch.sort(X[..., :num_samples, :], dim=-2).values,
                                )
                            )


class TestBoltzmannSampling(BotorchTestCase):
    def test_init(self):
        NO = "botorch.utils.testing.MockModel.num_outputs"
        with mock.patch(NO, new_callable=mock.PropertyMock) as mock_num_outputs:
            mock_num_outputs.return_value = 1
            mm = MockModel(None)
            acqf = PosteriorMean(mm)
            BS = BoltzmannSampling(acqf)
            self.assertEqual(BS.acq_func, acqf)
            self.assertEqual(BS.eta, 1.0)
            self.assertTrue(BS.replacement)
            BS = BoltzmannSampling(acqf, eta=0.5, replacement=False)
            self.assertEqual(BS.acq_func, acqf)
            self.assertEqual(BS.eta, 0.5)
            self.assertFalse(BS.replacement)

    def test_boltzmann_sampling(self):
        dtypes = (torch.float, torch.double)
        batch_shapes = (torch.Size(), torch.Size([3]))

        # test a bunch of combinations
        for batch_shape, N, d, num_samples, repl, dtype in itertools.product(
            batch_shapes, [6, 7], [1, 2], [4, 5], [True, False], dtypes
        ):
            tkwargs = {"device": self.device, "dtype": dtype}
            X = torch.rand(*batch_shape, N, d, **tkwargs)
            acqval = torch.randn(N, *batch_shape, **tkwargs)
            acqf = mock.Mock(return_value=acqval)
            BS = BoltzmannSampling(acqf, replacement=repl, eta=2.0)
            samples = BS(X, num_samples=num_samples)
            self.assertEqual(samples.shape, batch_shape + torch.Size([num_samples, d]))
            self.assertEqual(samples.dtype, dtype)
            if not repl:
                # check that we don't repeat points
                self.assertEqual(torch.unique(samples, dim=-2).size(-2), num_samples)

        # check that we do indeed pick the maximum for large eta
        for N, d, dtype in itertools.product([6, 7], [1, 2], dtypes):
            tkwargs = {"device": self.device, "dtype": dtype}
            X = torch.rand(N, d, **tkwargs)
            acqval = torch.zeros(N, **tkwargs)
            max_idx = torch.randint(N, (1,))
            acqval[max_idx] = 10.0
            acqf = mock.Mock(return_value=acqval)
            BS = BoltzmannSampling(acqf, eta=10.0)
            samples = BS(X, num_samples=1)
            self.assertTrue(torch.equal(samples, X[max_idx, :]))
