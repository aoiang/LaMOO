#! /usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
from botorch.exceptions.errors import BotorchError, BotorchTensorDimensionError
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.testing import BotorchTestCase


class TestNonDominatedPartitioning(BotorchTestCase):
    def test_non_dominated_partitioning(self):
        tkwargs = {"device": self.device}
        for dtype in (torch.float, torch.double):
            tkwargs["dtype"] = dtype
            ref_point = torch.zeros(2, **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=ref_point)
            # assert error is raised if pareto_Y has not been computed
            with self.assertRaises(BotorchError):
                partitioning.pareto_Y
            partitioning = NondominatedPartitioning(ref_point=ref_point)
            # test _update_pareto_Y
            Y = torch.ones(1, 2, **tkwargs)
            partitioning.update(Y=Y)

            partitioning._neg_Y = -Y
            partitioning.batch_shape = torch.Size([])
            self.assertFalse(partitioning._update_pareto_Y())

            # test m=2
            arange = torch.arange(3, 9, **tkwargs)
            pareto_Y = torch.stack([arange, 11 - arange], dim=-1)
            Y = torch.cat(
                [
                    pareto_Y,
                    torch.tensor(
                        [[8.0, 2.0], [7.0, 1.0]], **tkwargs
                    ),  # add some non-pareto elements
                ],
                dim=0,
            )
            partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y)
            sorting = torch.argsort(pareto_Y[:, 0], descending=True)
            self.assertTrue(torch.equal(pareto_Y[sorting], partitioning.pareto_Y))
            inf = float("inf")
            expected_cell_bounds = torch.tensor(
                [
                    [
                        [8.0, 0.0],
                        [7.0, 3.0],
                        [6.0, 4.0],
                        [5.0, 5.0],
                        [4.0, 6.0],
                        [3.0, 7.0],
                        [0.0, 8.0],
                    ],
                    [
                        [inf, inf],
                        [8.0, inf],
                        [7.0, inf],
                        [6.0, inf],
                        [5.0, inf],
                        [4.0, inf],
                        [3.0, inf],
                    ],
                ],
                **tkwargs,
            )
            cell_bounds = partitioning.get_hypercell_bounds()
            self.assertTrue(torch.equal(cell_bounds, expected_cell_bounds))
            # test compute hypervolume
            hv = partitioning.compute_hypervolume()
            self.assertEqual(hv.item(), 49.0)
            # test no pareto points better than the reference point
            partitioning = NondominatedPartitioning(
                ref_point=pareto_Y.max(dim=-2).values + 1, Y=Y
            )
            self.assertTrue(torch.equal(partitioning.pareto_Y, Y[:0]))
            self.assertEqual(partitioning.compute_hypervolume().item(), 0)

            Y = torch.rand(3, 10, 2, **tkwargs)
            # test batched m=2, no pareto points better than the reference point
            partitioning = NondominatedPartitioning(
                ref_point=Y.max(dim=-2).values + 1, Y=Y
            )
            self.assertTrue(torch.equal(partitioning.pareto_Y, Y[:, :0]))
            self.assertTrue(
                torch.equal(
                    partitioning.compute_hypervolume(),
                    torch.zeros(3, dtype=Y.dtype, device=Y.device),
                )
            )
            # test batched, m=2 basic
            partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y)
            cell_bounds = partitioning.get_hypercell_bounds()
            partitionings = []
            for i in range(Y.shape[0]):
                partitioning_i = NondominatedPartitioning(ref_point=ref_point, Y=Y[i])
                partitionings.append(partitioning_i)
                # check pareto_Y
                pareto_set1 = {tuple(x) for x in partitioning_i.pareto_Y.tolist()}
                pareto_set2 = {tuple(x) for x in partitioning.pareto_Y[i].tolist()}
                self.assertEqual(pareto_set1, pareto_set2)
                expected_cell_bounds_i = partitioning_i.get_hypercell_bounds()
                # remove padding
                no_padding_cell_bounds_i = cell_bounds[:, i][
                    :, ((cell_bounds[1, i] - cell_bounds[0, i]) != 0).all(dim=-1)
                ]
                self.assertTrue(
                    torch.equal(expected_cell_bounds_i, no_padding_cell_bounds_i)
                )

            # test improper Y shape (too many batch dims)
            with self.assertRaises(NotImplementedError):
                NondominatedPartitioning(ref_point=ref_point, Y=Y.unsqueeze(0))

            # test batched compute_hypervolume, m=2
            hvs = partitioning.compute_hypervolume()
            hvs_non_batch = torch.stack(
                [
                    partitioning_i.compute_hypervolume()
                    for partitioning_i in partitionings
                ],
                dim=0,
            )
            self.assertTrue(torch.allclose(hvs, hvs_non_batch))

            # test batched m>2
            ref_point = torch.zeros(3, **tkwargs)
            with self.assertRaises(NotImplementedError):
                NondominatedPartitioning(
                    ref_point=ref_point, Y=torch.cat([Y, Y[..., :1]], dim=-1)
                )

            # test error with partition_space_2d for m=3
            partitioning = NondominatedPartitioning(
                ref_point=ref_point, Y=torch.ones(1, 3, **tkwargs)
            )
            with self.assertRaises(BotorchTensorDimensionError):
                partitioning.partition_space_2d()
            # test m=3
            pareto_Y = torch.tensor(
                [[1.0, 6.0, 8.0], [2.0, 4.0, 10.0], [3.0, 5.0, 7.0]], **tkwargs
            )
            ref_point = torch.tensor([-1.0, -2.0, -3.0], **tkwargs)
            partitioning = NondominatedPartitioning(ref_point=ref_point, Y=pareto_Y)
            sorting = torch.argsort(pareto_Y[:, 0], descending=True)
            self.assertTrue(torch.equal(pareto_Y[sorting], partitioning.pareto_Y))

            expected_cell_bounds = torch.tensor(
                [
                    [
                        [1.0, 4.0, 7.0],
                        [-1.0, -2.0, 10.0],
                        [-1.0, 4.0, 8.0],
                        [1.0, -2.0, 10.0],
                        [1.0, 4.0, 8.0],
                        [-1.0, 6.0, -3.0],
                        [1.0, 5.0, -3.0],
                        [-1.0, 5.0, 8.0],
                        [2.0, -2.0, 7.0],
                        [2.0, 4.0, 7.0],
                        [3.0, -2.0, -3.0],
                        [2.0, -2.0, 8.0],
                        [2.0, 5.0, -3.0],
                    ],
                    [
                        [2.0, 5.0, 8.0],
                        [1.0, 4.0, inf],
                        [1.0, 5.0, inf],
                        [2.0, 4.0, inf],
                        [2.0, 5.0, inf],
                        [1.0, inf, 8.0],
                        [2.0, inf, 8.0],
                        [2.0, inf, inf],
                        [3.0, 4.0, 8.0],
                        [3.0, 5.0, 8.0],
                        [inf, 5.0, 8.0],
                        [inf, 5.0, inf],
                        [inf, inf, inf],
                    ],
                ],
                **tkwargs,
            )
            cell_bounds = partitioning.get_hypercell_bounds()
            # cell bounds can have different order
            num_matches = (
                (cell_bounds.unsqueeze(0) == expected_cell_bounds.unsqueeze(1))
                .all(dim=-1)
                .any(dim=0)
                .sum()
            )
            self.assertTrue(num_matches, 9)
            # test compute hypervolume
            hv = partitioning.compute_hypervolume()
            self.assertEqual(hv.item(), 358.0)

            # test no pareto points better than the reference point
            partitioning = NondominatedPartitioning(
                ref_point=pareto_Y.max(dim=-2).values + 1, Y=pareto_Y
            )
            self.assertTrue(torch.equal(partitioning.pareto_Y, pareto_Y[:0]))
            self.assertEqual(partitioning.compute_hypervolume().item(), 0)

            # TODO: test approximate decomposition
