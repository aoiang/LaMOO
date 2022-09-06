#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.acquisition.multi_objective.analytic import (
    ExpectedHypervolumeImprovement,
    MultiObjectiveAnalyticAcquisitionFunction,
)
from botorch.acquisition.multi_objective.objective import (
    AnalyticMultiOutputObjective,
    IdentityAnalyticMultiOutputObjective,
    IdentityMCMultiOutputObjective,
)
from botorch.exceptions.errors import BotorchError, UnsupportedError
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase, MockModel, MockPosterior


class DummyMultiObjectiveAnalyticAcquisitionFunction(
    MultiObjectiveAnalyticAcquisitionFunction
):
    def forward(self, X):
        pass


class DummyAnalyticMultiOutputObjective(AnalyticMultiOutputObjective):
    def forward(self, samples):
        pass


class TestMultiObjectiveAnalyticAcquisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            MultiObjectiveAnalyticAcquisitionFunction()

    def test_init(self):
        mm = MockModel(MockPosterior(mean=torch.rand(2, 1)))
        # test default init
        acqf = DummyMultiObjectiveAnalyticAcquisitionFunction(model=mm)
        self.assertIsInstance(acqf.objective, IdentityAnalyticMultiOutputObjective)
        # test custom init
        objective = DummyAnalyticMultiOutputObjective()
        acqf = DummyMultiObjectiveAnalyticAcquisitionFunction(
            model=mm, objective=objective
        )
        self.assertEqual(acqf.objective, objective)
        # test unsupported objective
        with self.assertRaises(UnsupportedError):
            DummyMultiObjectiveAnalyticAcquisitionFunction(
                model=mm, objective=IdentityMCMultiOutputObjective()
            )
        acqf = DummyMultiObjectiveAnalyticAcquisitionFunction(model=mm)
        # test set_X_pending
        with self.assertRaises(UnsupportedError):
            acqf.set_X_pending()


class TestExpectedHypervolumeImprovement(BotorchTestCase):
    def test_expected_hypervolume_improvement(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            ref_point = [0.0, 0.0]
            tkwargs["dtype"] = dtype
            pareto_Y = torch.tensor(
                [[4.0, 5.0], [5.0, 5.0], [8.5, 3.5], [8.5, 3.0], [9.0, 1.0]], **tkwargs
            )
            partitioning = NondominatedPartitioning(
                ref_point=torch.tensor(ref_point, **tkwargs)
            )
            # the event shape is `b x q x m` = 1 x 1 x 1
            mean = torch.zeros(1, 1, 2, **tkwargs)
            variance = torch.zeros(1, 1, 2, **tkwargs)
            mm = MockModel(MockPosterior(mean=mean, variance=variance))
            # test error if there is not pareto_Y initialized in partitioning
            with self.assertRaises(BotorchError):
                ExpectedHypervolumeImprovement(
                    model=mm, ref_point=ref_point, partitioning=partitioning
                )
            partitioning.update(Y=pareto_Y)
            # test error if ref point has wrong shape
            with self.assertRaises(ValueError):
                ExpectedHypervolumeImprovement(
                    model=mm, ref_point=ref_point[:1], partitioning=partitioning
                )

            with self.assertRaises(ValueError):
                # test error if no pareto_Y point is better than ref_point
                ExpectedHypervolumeImprovement(
                    model=mm, ref_point=[10.0, 10.0], partitioning=partitioning
                )
            X = torch.zeros(1, 1, **tkwargs)
            # basic test
            acqf = ExpectedHypervolumeImprovement(
                model=mm, ref_point=ref_point, partitioning=partitioning
            )
            res = acqf(X)
            self.assertEqual(res.item(), 0.0)
            # check ref point
            self.assertTrue(
                torch.equal(acqf.ref_point, torch.tensor(ref_point, **tkwargs))
            )
            # check bounds
            self.assertTrue(hasattr(acqf, "cell_lower_bounds"))
            self.assertTrue(hasattr(acqf, "cell_upper_bounds"))
            # check cached indices
            expected_indices = torch.tensor(
                [[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long, device=self.device
            )
            self.assertTrue(torch.equal(acqf._cross_product_indices, expected_indices))
