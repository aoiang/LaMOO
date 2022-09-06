#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Algorithms for partitioning the non-dominated space into rectangles.

References

.. [Couckuyt2012]
    I. Couckuyt, D. Deschrijver and T. Dhaene, "Towards Efficient
    Multiobjective Optimization: Multiobjective statistical criterions,"
    2012 IEEE Congress on Evolutionary Computation, Brisbane, QLD, 2012,
    pp. 1-8.

"""

from __future__ import annotations

from typing import Optional

import torch
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch.utils.multi_objective.box_decompositions.box_decomposition import (
    BoxDecomposition,
)
from botorch.utils.multi_objective.box_decompositions.utils import _expand_ref_point
from torch import Tensor


class NondominatedPartitioning(BoxDecomposition):
    r"""A class for partitioning the non-dominated space into hyper-cells.

    Note: this assumes maximization. Internally, it multiplies by -1 and performs
    the decomposition under minimization. TODO: use maximization internally as well.

    Note: it is only feasible to use this algorithm to compute an exact
    decomposition of the non-dominated space for `m<5` objectives (alpha=0.0).

    The alpha parameter can be increased to obtain an approximate partitioning
    faster. The `alpha` is a fraction of the total hypervolume encapsuling the
    entire Pareto set. When a hypercell's volume divided by the total hypervolume
    is less than `alpha`, we discard the hypercell. See Figure 2 in
    [Couckuyt2012]_ for a visual representation.

    This PyTorch implementation of the binary partitioning algorithm ([Couckuyt2012]_)
    is adapted from numpy/tensorflow implementation at:
    https://github.com/GPflow/GPflowOpt/blob/master/gpflowopt/pareto.py.

    TODO: replace this with a more efficient decomposition. E.g.
    https://link.springer.com/content/pdf/10.1007/s10898-019-00798-7.pdf
    """

    def __init__(
        self,
        ref_point: Tensor,
        Y: Optional[Tensor] = None,
        alpha: float = 0.0,
    ) -> None:
        """Initialize NondominatedPartitioning.

        Args:
            ref_point: A `m`-dim tensor containing the reference point.
            Y: A `(batch_shape) x n x m`-dim tensor.
            alpha: A thresold fraction of total volume used in an approximate
                decomposition.
        """
        self.alpha = alpha
        super().__init__(ref_point=ref_point, sort=True, Y=Y)

    def _partition_space(self) -> None:
        r"""Partition the non-dominated space into disjoint hypercells.

        This method works for an arbitrary number of outcomes, but is
        less efficient than `partition_non_dominated_space_2d` for the
        2-outcome case.
        """
        # The binary parititoning algorithm uses indices the augmented Pareto front.
        # n_pareto + 2 x m
        aug_pareto_Y_idcs = self._get_augmented_pareto_front_indices()

        # Initialize one cell over entire pareto front
        cell = torch.zeros(
            2, self.num_outcomes, dtype=torch.long, device=self._neg_Y.device
        )
        cell[1] = aug_pareto_Y_idcs.shape[0] - 1
        stack = [cell]

        # hypercells contains the indices of the (augmented) Pareto front
        # that specify that bounds of the each hypercell.
        # It is a `2 x num_cells x num_outcomes`-dim tensor
        self.register_buffer(
            "hypercells",
            torch.empty(
                2, 0, self.num_outcomes, dtype=torch.long, device=self._neg_Y.device
            ),
        )
        outcome_idxr = torch.arange(
            self.num_outcomes, dtype=torch.long, device=self._neg_Y.device
        )

        # edge case: empty pareto set
        # use a single cell
        if self._neg_pareto_Y.shape[-2] == 0:
            # 2 x m
            cell_bounds_pareto_idcs = aug_pareto_Y_idcs[cell, outcome_idxr]
            self.hypercells = torch.cat(
                [self.hypercells, cell_bounds_pareto_idcs.unsqueeze(1)], dim=1
            )
        else:
            # Extend Pareto front with the ideal and anti-ideal point
            ideal_point = self._neg_pareto_Y.min(dim=0, keepdim=True).values - 1
            anti_ideal_point = self._neg_pareto_Y.max(dim=0, keepdim=True).values + 1
            # `n_pareto + 2 x m`
            aug_pareto_Y = torch.cat(
                [ideal_point, self._neg_pareto_Y, anti_ideal_point], dim=0
            )

            total_volume = (anti_ideal_point - ideal_point).prod()

            # Use binary partitioning
            while len(stack) > 0:
                # The following 3 tensors are all `2 x m`
                cell = stack.pop()
                cell_bounds_pareto_idcs = aug_pareto_Y_idcs[cell, outcome_idxr]
                cell_bounds_pareto_values = aug_pareto_Y[
                    cell_bounds_pareto_idcs, outcome_idxr
                ]
                # Check cell bounds
                # - if cell upper bound is better than Pareto front on all outcomes:
                #   - accept the cell
                # - elif cell lower bound is better than Pareto front on all outcomes:
                #   - this means the cell overlaps the Pareto front. Divide the cell
                #     along its longest edge.
                if (
                    (cell_bounds_pareto_values[1] <= self._neg_pareto_Y)
                    .any(dim=1)
                    .all()
                ):
                    # Cell is entirely non-dominated
                    self.hypercells = torch.cat(
                        [self.hypercells, cell_bounds_pareto_idcs.unsqueeze(1)], dim=1
                    )
                elif (
                    (cell_bounds_pareto_values[0] <= self._neg_pareto_Y)
                    .any(dim=1)
                    .all()
                ):
                    # The cell overlaps the pareto front
                    # compute the distance (in integer indices)
                    # This has shape `m`
                    idx_dist = cell[1] - cell[0]

                    any_not_adjacent = (idx_dist > 1).any()
                    cell_volume = (
                        (cell_bounds_pareto_values[1] - cell_bounds_pareto_values[0])
                        .prod(dim=-1)
                        .item()
                    )

                    # Only divide a cell when it is not composed of adjacent indices
                    # and the fraction of total volume is above the approximation
                    # threshold fraction
                    if (
                        any_not_adjacent
                        and ((cell_volume / total_volume) > self.alpha).all()
                    ):
                        # Divide the test cell over its largest dimension
                        # largest (by index length)
                        length, longest_dim = torch.max(idx_dist, dim=0)
                        length = length.item()
                        longest_dim = longest_dim.item()

                        new_length1 = int(round(length / 2.0))
                        new_length2 = length - new_length1

                        # Store divided cells
                        # cell 1: subtract new_length1 from the upper bound of the cell
                        # cell 2: add new_length2 to the lower bound of the cell
                        for bound_idx, length_delta in (
                            (1, -new_length1),
                            (0, new_length2),
                        ):
                            new_cell = cell.clone()
                            new_cell[bound_idx, longest_dim] += length_delta
                            stack.append(new_cell)

    def partition_space_2d(self) -> None:
        r"""Partition the non-dominated space into disjoint hypercells.

        This direct method works for `m=2` outcomes.
        """
        if self.num_outcomes != 2:
            raise BotorchTensorDimensionError(
                "partition_non_dominated_space_2d requires 2 outputs, "
                f"but num_outcomes={self.num_outcomes}"
            )
        pf_ext_idx = self._get_augmented_pareto_front_indices()
        n_pf_plus_1 = self._neg_pareto_Y.shape[-2] + 1
        view_shape = torch.Size([1] * len(self.batch_shape) + [n_pf_plus_1])
        expand_shape = self.batch_shape + torch.Size([n_pf_plus_1])
        range_pf_plus1 = torch.arange(
            n_pf_plus_1, dtype=torch.long, device=self._neg_pareto_Y.device
        )
        range_pf_plus1_expanded = range_pf_plus1.view(view_shape).expand(expand_shape)

        lower = torch.stack(
            [range_pf_plus1_expanded, torch.zeros_like(range_pf_plus1_expanded)], dim=-1
        )
        upper = torch.stack(
            [1 + range_pf_plus1_expanded, pf_ext_idx[..., -range_pf_plus1 - 1, -1]],
            dim=-1,
        )
        # 2 x batch_shape x n_cells x 2
        self.register_buffer("hypercells", torch.stack([lower, upper], dim=0))

    def _get_augmented_pareto_front_indices(self) -> Tensor:
        r"""Get indices of augmented Pareto front."""
        pf_idx = torch.argsort(self._neg_pareto_Y, dim=-2)
        return torch.cat(
            [
                torch.zeros(
                    *self.batch_shape,
                    1,
                    self.num_outcomes,
                    dtype=torch.long,
                    device=self._neg_Y.device,
                ),
                # Add 1 because index zero is used for the ideal point
                pf_idx + 1,
                torch.full(
                    torch.Size(
                        [
                            *self.batch_shape,
                            1,
                            self.num_outcomes,
                        ]
                    ),
                    self._neg_pareto_Y.shape[-2] + 1,
                    dtype=torch.long,
                    device=self._neg_Y.device,
                ),
            ],
            dim=-2,
        )

    def get_hypercell_bounds(self) -> Tensor:
        r"""Get the bounds of each hypercell in the decomposition.

        Args:
            ref_point: A `(batch_shape) x m`-dim tensor containing the reference point.

        Returns:
            A `2 x num_cells x num_outcomes`-dim tensor containing the
                lower and upper vertices bounding each hypercell.
        """
        ref_point = _expand_ref_point(
            ref_point=self.ref_point, batch_shape=self.batch_shape
        )
        aug_pareto_Y = torch.cat(
            [
                # -inf is the lower bound of the non-dominated space
                torch.full(
                    torch.Size(
                        [
                            *self.batch_shape,
                            1,
                            self.num_outcomes,
                        ]
                    ),
                    float("-inf"),
                    dtype=self._neg_pareto_Y.dtype,
                    device=self._neg_pareto_Y.device,
                ),
                self._neg_pareto_Y,
                # note: internally, this class minimizes, so use negative here
                -(ref_point.unsqueeze(-2)),
            ],
            dim=-2,
        )
        minimization_cell_bounds = self._get_hypercell_bounds(aug_pareto_Y=aug_pareto_Y)
        # swap upper and lower bounds and multiply by -1
        return -minimization_cell_bounds.flip(0)

    def _get_hypercell_bounds(self, aug_pareto_Y: Tensor) -> Tensor:
        r"""Get the bounds of each hypercell in the decomposition.

        Args:
            aug_pareto_Y: A `n_pareto + 2 x m`-dim tensor containing
            the augmented Pareto front.

        Returns:
            A `2 x (batch_shape) x num_cells x num_outcomes`-dim tensor containing the
                lower and upper vertices bounding each hypercell.
        """
        num_cells = self.hypercells.shape[-2]
        cells_times_outcomes = num_cells * self.num_outcomes
        outcome_idxr = (
            torch.arange(self.num_outcomes, dtype=torch.long, device=self._neg_Y.device)
            .repeat(num_cells)
            .view(
                *(1 for _ in self.hypercells.shape[:-2]),
                cells_times_outcomes,
            )
            .expand(*self.hypercells.shape[:-2], cells_times_outcomes)
        )

        # this tensor is 2 x (num_cells * num_outcomes) x 2
        # the batch dim corresponds to lower/upper bound
        cell_bounds_idxr = torch.stack(
            [
                self.hypercells.view(*self.hypercells.shape[:-2], -1),
                outcome_idxr,
            ],
            dim=-1,
        ).view(2, -1, 2)
        if len(self.batch_shape) > 0:
            # TODO: support multiple batch dimensions here
            batch_idxr = (
                torch.arange(
                    self.batch_shape[0], dtype=torch.long, device=self._neg_Y.device
                )
                .unsqueeze(1)
                .expand(-1, cells_times_outcomes)
                .reshape(1, -1, 1)
                .expand(2, -1, 1)
            )
            cell_bounds_idxr = torch.cat([batch_idxr, cell_bounds_idxr], dim=-1)

        cell_bounds_values = aug_pareto_Y[
            cell_bounds_idxr.chunk(cell_bounds_idxr.shape[-1], dim=-1)
        ]
        view_shape = (2, *self.batch_shape, num_cells, self.num_outcomes)
        return cell_bounds_values.view(view_shape)

    def compute_hypervolume(self) -> Tensor:
        r"""Compute the hypervolume for the given reference point.

        Note: This assumes minimization.

        This method computes the hypervolume of the non-dominated space
        and computes the difference between the hypervolume between the
        ideal point and hypervolume of the non-dominated space.

        Note there are much more efficient alternatives for computing
        hypervolume when m > 2 (which do not require partitioning the
        non-dominated space). Given such a partitioning, this method
        is quite fast.

        Returns:
            `(batch_shape)`-dim tensor containing the dominated hypervolume.
        """
        if self._neg_pareto_Y.shape[-2] == 0:
            return torch.zeros(
                self._neg_pareto_Y.shape[:-2],
                dtype=self._neg_pareto_Y.dtype,
                device=self._neg_pareto_Y.device,
            )
        ref_point = _expand_ref_point(
            ref_point=self.ref_point, batch_shape=self.batch_shape
        )
        # internally we minimize
        ref_point = -ref_point.unsqueeze(-2)
        ideal_point = self._neg_pareto_Y.min(dim=-2, keepdim=True).values
        aug_pareto_Y = torch.cat([ideal_point, self._neg_pareto_Y, ref_point], dim=-2)
        cell_bounds_values = self._get_hypercell_bounds(aug_pareto_Y=aug_pareto_Y)
        total_volume = (ref_point - ideal_point).squeeze(-2).prod(dim=-1)
        non_dom_volume = (
            (cell_bounds_values[1] - cell_bounds_values[0]).prod(dim=-1).sum(dim=-1)
        )
        return total_volume - non_dom_volume
