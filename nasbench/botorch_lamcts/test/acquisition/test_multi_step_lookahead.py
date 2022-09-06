#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from botorch.acquisition import (
    qMultiStepLookahead,
    qExpectedImprovement,
    ExpectedImprovement,
)
from botorch.acquisition.multi_step_lookahead import make_best_f, warmstart_multistep
from botorch.acquisition.objective import IdentityMCObjective, ScalarizedObjective
from botorch.exceptions.errors import UnsupportedError
from botorch.models import SingleTaskGP
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.testing import BotorchTestCase


class TestMultiStepLookahead(BotorchTestCase):
    def test_qMS_init(self):
        d = 2
        q = 1
        num_data = 3
        q_batch_sizes = [1, 1, 1]
        num_fantasies = [2, 2, 1]
        t_batch_size = [2]
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0], [1]], device=self.device, dtype=dtype)
            bounds = bounds.repeat(1, d)
            train_X = torch.rand(num_data, d, device=self.device, dtype=dtype)
            train_Y = torch.rand(num_data, 1, device=self.device, dtype=dtype)
            model = SingleTaskGP(train_X, train_Y)

            # exactly one of samplers or num_fantasies
            with self.assertRaises(UnsupportedError):
                qMultiStepLookahead(
                    model=model,
                    batch_sizes=q_batch_sizes,
                    valfunc_cls=[qExpectedImprovement] * 4,
                    valfunc_argfacs=[make_best_f] * 4,
                    inner_mc_samples=[2] * 4,
                )

            # cannot use qMS as its own valfunc_cls
            with self.assertRaises(UnsupportedError):
                qMultiStepLookahead(
                    model=model,
                    batch_sizes=q_batch_sizes,
                    valfunc_cls=[qMultiStepLookahead] * 4,
                    valfunc_argfacs=[make_best_f] * 4,
                    num_fantasies=num_fantasies,
                    inner_mc_samples=[2] * 4,
                )

            # construct using samplers
            samplers = [
                SobolQMCNormalSampler(
                    num_samples=nf, resample=False, collapse_batch_dims=True
                )
                for nf in num_fantasies
            ]
            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=q_batch_sizes,
                valfunc_cls=[qExpectedImprovement] * 4,
                valfunc_argfacs=[make_best_f] * 4,
                inner_mc_samples=[2] * 4,
                samplers=samplers,
            )
            self.assertEqual(qMS.num_fantasies, num_fantasies)

            # use default valfunc_cls, valfun_argfacs, inner_mc_samples
            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=q_batch_sizes,
                samplers=samplers,
            )
            self.assertEqual(len(qMS._valfunc_cls), 4)
            self.assertEqual(len(qMS.inner_samplers), 4)
            self.assertEqual(len(qMS._valfunc_argfacs), 4)

            # _construct_inner_samplers error catching tests below
            # AnalyticAcquisitionFunction with MCAcquisitionObjective
            with self.assertRaises(UnsupportedError):
                qMultiStepLookahead(
                    model=model,
                    objective=IdentityMCObjective(),
                    batch_sizes=q_batch_sizes,
                    valfunc_cls=[ExpectedImprovement] * 4,
                    valfunc_argfacs=[make_best_f] * 4,
                    num_fantasies=num_fantasies,
                )
            # AnalyticAcquisitionFunction and q > 1
            with self.assertRaises(UnsupportedError):
                qMultiStepLookahead(
                    model=model,
                    batch_sizes=[2, 2, 2],
                    valfunc_cls=[ExpectedImprovement] * 4,
                    valfunc_argfacs=[make_best_f] * 4,
                    num_fantasies=num_fantasies,
                    inner_mc_samples=[2] * 4,
                )
            # AnalyticAcquisitionFunction and inner_mc_samples
            with self.assertWarns(Warning):
                qMultiStepLookahead(
                    model=model,
                    batch_sizes=q_batch_sizes,
                    valfunc_cls=[ExpectedImprovement] * 4,
                    valfunc_argfacs=[make_best_f] * 4,
                    num_fantasies=num_fantasies,
                    inner_mc_samples=[2] * 4,
                )
            # MCAcquisitionFunction and non MCAcquisitionObjective
            with self.assertRaises(UnsupportedError):
                qMultiStepLookahead(
                    model=model,
                    objective=ScalarizedObjective(weights=torch.tensor([1.0])),
                    batch_sizes=[2, 2, 2],
                    valfunc_cls=[qExpectedImprovement] * 4,
                    valfunc_argfacs=[make_best_f] * 4,
                    num_fantasies=num_fantasies,
                    inner_mc_samples=[2] * 4,
                )

            # test warmstarting
            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=q_batch_sizes,
                samplers=samplers,
            )
            q_prime = qMS.get_augmented_q_batch_size(q)
            eval_X = torch.rand(t_batch_size + [q_prime, d])
            warmstarted_X = warmstart_multistep(
                acq_function=qMS,
                bounds=bounds,
                num_restarts=5,
                raw_samples=10,
                full_optimizer=eval_X,
            )
            self.assertEquals(warmstarted_X.shape, torch.Size([5, q_prime, d]))

    def test_qMS(self):
        d = 2
        q = 1
        num_data = 3
        q_batch_sizes = [1, 1, 1]
        num_fantasies = [2, 2, 1]
        t_batch_size = [2]
        for dtype in (torch.float, torch.double):
            bounds = torch.tensor([[0], [1]], device=self.device, dtype=dtype)
            bounds = bounds.repeat(1, d)
            train_X = torch.rand(num_data, d, device=self.device, dtype=dtype)
            train_Y = torch.rand(num_data, 1, device=self.device, dtype=dtype)
            model = SingleTaskGP(train_X, train_Y)

            # default evaluation tests
            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=[1, 1, 1],
                num_fantasies=num_fantasies,
            )
            q_prime = qMS.get_augmented_q_batch_size(q)
            eval_X = torch.rand(t_batch_size + [q_prime, d])
            result = qMS(eval_X)
            self.assertEqual(result.shape, torch.Size(t_batch_size))

            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=q_batch_sizes,
                valfunc_cls=[qExpectedImprovement] * 4,
                valfunc_argfacs=[make_best_f] * 4,
                num_fantasies=num_fantasies,
                inner_mc_samples=[2] * 4,
            )
            result = qMS(eval_X)
            self.assertEqual(result.shape, torch.Size(t_batch_size))

            # get induced fantasy model, with collapse_fantasy_base_samples
            fant_model = qMS.get_induced_fantasy_model(eval_X)
            self.assertEqual(
                fant_model.train_inputs[0].shape,
                torch.Size(
                    num_fantasies[::-1]
                    + t_batch_size
                    + [num_data + sum(q_batch_sizes), d]
                ),
            )

            # collapse fantasy base samples
            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=q_batch_sizes,
                valfunc_cls=[qExpectedImprovement] * 4,
                valfunc_argfacs=[make_best_f] * 4,
                num_fantasies=num_fantasies,
                inner_mc_samples=[2] * 4,
                collapse_fantasy_base_samples=False,
            )
            q_prime = qMS.get_augmented_q_batch_size(q)
            eval_X = torch.rand(t_batch_size + [q_prime, d])
            result = qMS(eval_X)
            self.assertEqual(result.shape, torch.Size(t_batch_size))
            self.assertEqual(qMS.samplers[0].batch_range, (-3, -2))

            # get induced fantasy model, without collapse_fantasy_base_samples
            fant_model = qMS.get_induced_fantasy_model(eval_X)
            self.assertEqual(
                fant_model.train_inputs[0].shape,
                torch.Size(
                    num_fantasies[::-1]
                    + t_batch_size
                    + [num_data + sum(q_batch_sizes), d]
                ),
            )

            # X_pending
            X_pending = torch.rand(5, d)
            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=q_batch_sizes,
                valfunc_cls=[qExpectedImprovement] * 4,
                valfunc_argfacs=[make_best_f] * 4,
                num_fantasies=num_fantasies,
                inner_mc_samples=[2] * 4,
                X_pending=X_pending,
            )
            q_prime = qMS.get_augmented_q_batch_size(q)
            eval_X = torch.rand(t_batch_size + [q_prime, d])
            result = qMS(eval_X)
            self.assertEqual(result.shape, torch.Size(t_batch_size))

            # add dummy base_weights to samplers
            samplers = [
                SobolQMCNormalSampler(
                    num_samples=nf, resample=False, collapse_batch_dims=True
                )
                for nf in num_fantasies
            ]
            for s in samplers:
                s.base_weights = torch.ones(
                    s.sample_shape[0], 1, device=self.device, dtype=dtype
                )

            qMS = qMultiStepLookahead(
                model=model,
                batch_sizes=[1, 1, 1],
                samplers=samplers,
            )
            q_prime = qMS.get_augmented_q_batch_size(q)
            eval_X = torch.rand(t_batch_size + [q_prime, d])
            result = qMS(eval_X)
            self.assertEqual(result.shape, torch.Size(t_batch_size))

            # extract candidates
            cand = qMS.extract_candidates(eval_X)
            self.assertEqual(cand.shape, torch.Size(t_batch_size + [q, d]))
