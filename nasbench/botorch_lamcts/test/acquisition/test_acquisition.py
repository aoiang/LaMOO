#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.utils.testing import BotorchTestCase


class TestAcquisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            AcquisitionFunction()


class TestOneShotAcquisitionFunction(BotorchTestCase):
    def test_abstract_raises(self):
        with self.assertRaises(TypeError):
            OneShotAcquisitionFunction()
