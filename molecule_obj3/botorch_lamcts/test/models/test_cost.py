#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from random import random

import torch
from botorch.models.cost import AffineFidelityCostModel
from botorch.utils.testing import BotorchTestCase


class TestCostModels(BotorchTestCase):
    def test_affine_fidelity_cost_model(self):
        for dtype in (torch.float, torch.double):
            for batch_shape in ([], [2]):
                X = torch.rand(*batch_shape, 3, 4, device=self.device, dtype=dtype)
                # test default parameters
                model = AffineFidelityCostModel()
                self.assertEqual(model.num_outputs, 1)
                self.assertEqual(model.fidelity_dims, [-1])
                self.assertEqual(model.fixed_cost, 0.01)
                cost = model(X)
                cost_exp = 0.01 + X[..., -1:]
                self.assertTrue(torch.allclose(cost, cost_exp))
                # test custom parameters
                fw = {2: 2.0, 0: 1.0}
                fc = random()
                model = AffineFidelityCostModel(fidelity_weights=fw, fixed_cost=fc)
                self.assertEqual(model.fidelity_dims, [0, 2])
                self.assertEqual(model.fixed_cost, fc)
                cost = model(X)
                cost_exp = fc + sum(v * X[..., i : i + 1] for i, v in fw.items())
                self.assertTrue(torch.allclose(cost, cost_exp))
