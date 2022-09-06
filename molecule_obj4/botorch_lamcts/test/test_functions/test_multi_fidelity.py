#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from botorch.test_functions.multi_fidelity import (
    AugmentedBranin,
    AugmentedHartmann,
    AugmentedRosenbrock,
)
from botorch.utils.testing import BotorchTestCase, SyntheticTestFunctionBaseTestCase


class TestAugmentedBranin(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [
        AugmentedBranin(),
        AugmentedBranin(negate=True),
        AugmentedBranin(noise_std=0.1),
    ]


class TestAugmentedHartmann(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [
        AugmentedHartmann(),
        AugmentedHartmann(negate=True),
        AugmentedHartmann(noise_std=0.1),
    ]


class TestAugmentedRosenbrock(SyntheticTestFunctionBaseTestCase, BotorchTestCase):

    functions = [
        AugmentedRosenbrock(),
        AugmentedRosenbrock(negate=True),
        AugmentedRosenbrock(noise_std=0.1),
        AugmentedRosenbrock(dim=4),
        AugmentedRosenbrock(dim=4, negate=True),
        AugmentedRosenbrock(dim=4, noise_std=0.1),
    ]

    def test_min_dimension(self):
        with self.assertRaises(ValueError):
            AugmentedRosenbrock(dim=2)
