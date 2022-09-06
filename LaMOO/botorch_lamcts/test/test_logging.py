#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from botorch import settings
from botorch.logging import LOG_LEVEL_DEFAULT, logger
from botorch.utils.testing import BotorchTestCase


class TestLogging(BotorchTestCase):
    def test_logger(self):
        # Verify log statements are properly captured
        # assertLogs() captures all log calls, ignoring the severity level
        with self.assertLogs(logger="botorch", level="INFO") as logs_cm:
            logger.info("Hello World!")
            logger.error("Goodbye Universe!")
        self.assertEqual(
            logs_cm.output,
            ["INFO:botorch:Hello World!", "ERROR:botorch:Goodbye Universe!"],
        )

    def test_settings_log_level(self):
        # Verify the default level is applied
        self.assertEqual(logger.level, LOG_LEVEL_DEFAULT)
        # Next, verify the level of overwritten within the context manager
        with settings.log_level(logging.INFO):
            self.assertEqual(logger.level, logging.INFO)
        # Finally, verify the original level is set again
        self.assertEqual(logger.level, LOG_LEVEL_DEFAULT)
