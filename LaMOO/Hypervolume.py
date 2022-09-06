#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

r"""Hypervolume Utilities.

References

.. [Fonseca2006]
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.

"""

# from __future__ import annotations

from typing import Optional, Tuple

import torch
# from botorch.exceptions.errors import BotorchTensorDimensionError
from torch import Tensor
import numpy as np

def calculate_pareto(Y: Tensor, maximize: bool = True):

    is_efficient = torch.ones(*Y.shape[:-1], dtype=bool, device=Y.device)
    for i in range(Y.shape[-2]):
        vals = Y[..., i : i + 1, :]
        if maximize:
            update = torch.any(Y >= vals, dim=-1)[is_efficient]
        else:
            update = torch.any(Y <= vals, dim=-1)[is_efficient]
        is_efficient[is_efficient] = update
    return is_efficient


def get_pareto(X: Tensor, Y: Tensor, maximize: bool = True):

    pareto_mask = calculate_pareto(Y, maximize=maximize)
    return X[pareto_mask], Y[pareto_mask]

def pareto_sort(pareto_Y: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Sort 2 objectives in non-decreasing and non-increasing order respectively.

    Args:
        pareto_Y: a `(batch_shape) x n_pareto x 2`-dim tensor of pareto outcomes

    Returns:
        2-element tuple containing

        - A `(batch_shape) x n_pareto x 2`-dim tensor of sorted values
        - A `(batch_shape) x n_pareto`-dim tensor of indices
    """
    if pareto_Y.shape[-1] != 2:
        raise NotImplementedError(
            f"There must be exactly 2 objectives, got {pareto_Y.shape[-1]}."
        )
    # sort by second outcome
    # this tensor is batch_shape x n
    inner_sorting = torch.argsort(pareto_Y[..., 1], descending=True)
    # expand to batch_shape x n x 2
    inner_sorting = inner_sorting.unsqueeze(-1).expand(
        *inner_sorting.shape, pareto_Y.shape[-1]
    )
    pareto_Y_inner_sorted = pareto_Y.gather(-2, inner_sorting)
    # this tensor is batch_shape x n
    # TODO: replace numpy stable sorting https://github.com/pytorch/pytorch/issues/28871
    outer_sorting = torch.from_numpy(
        np.argsort(pareto_Y_inner_sorted[..., 0].cpu().numpy(), kind="stable", axis=-1)
    ).to(device=pareto_Y.device)
    # expand to batch_shape x n x 2
    outer_sorting = outer_sorting.unsqueeze(-1).expand(
        *outer_sorting.shape, pareto_Y.shape[-1]
    )
    values = pareto_Y_inner_sorted.gather(-2, outer_sorting)
    indices = inner_sorting[..., 0].gather(dim=-1, index=outer_sorting[..., 0])
    return values, indices


def compute_hypervolume_2d(
    pareto_Y: Tensor, ref_point: Tensor, compute_contributions: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Compute hypervolume (and contributions) for a two-objective pareto front.

    This method assumes maximization and requires that all `pareto_Y` points
    are greater than or equal to the reference point.

    TODO: Make this function require sorted inputs (so that we don't re-sort and
    re-invert the sorting with every HV computation): T65679256

    Note: the HV contributions depend on the pareto front that is provided. If a
    weakly pareto font is passed in, the hypervolume contributions will be with with
    respect to the other points on pareto and differentthan if a non-dominated front
    were provided.

    Args:
        pareto_Y: a `(batch_shape) x n_pareto x 2`-dim tensor of pareto outcomes
        ref_point: a `2`-dim reference point
        compute_contributions: boolean indicating whether to compute hypervolume
            contributions for each pareto point.

    Returns:
        2-element tuple containing

        - A `(batch_shape)`-dim tensor of hypervolumes.
        - An optional `(batch_shape) x n_pareto`-dim tensor of hypervolume contributions

    """
    if pareto_Y.shape[-1] != 2 or ref_point.shape[-1] != 2:
        raise NotImplementedError("Hypervolume is only supported for 2 objectives.")
    elif ref_point.ndim > 1 and pareto_Y.shape[:-2] != ref_point.shape[:-1]:
        raise BotorchTensorDimensionError(
            "ref_point must either have no batch shape or the same batch shape "
            f"as pareto_Y. Got {ref_point.shape[:-1]}, but expected "
            f"{pareto_Y.shape[:-2]} or torch.Size([])."
        )
    elif (pareto_Y <= ref_point.unsqueeze(-2)).any():

        valid_points = []
        # print('cur_pareto_Y is', pareto_Y)
        # print('unsqueezed ref_point is', ref_point.unsqueeze(-2))
        for point in (pareto_Y > ref_point.unsqueeze(-2)):
            valid_points.append(point.all())
        # print('new pareto_Y is', (pareto_Y > ref_point.unsqueeze(-2)))

        valid_points = torch.tensor(valid_points)
        # print('new pareto_Y2 is', pareto_Y[valid_points])
        pareto_Y = pareto_Y[valid_points]
        if len(pareto_Y) == 0:
            return torch.tensor(0.0), None
        # raise ValueError("All pareto_Y objectives must be greater than the ref_point.")

    pareto_Y_sorted, sorting_indices = pareto_sort(pareto_Y)
    # add boundary point to each front
    # the boundary point is the extreme value in each outcome
    # (a single coordinate of reference point)
    batch_shape = pareto_Y_sorted.shape[:-2]
    if ref_point.ndim == pareto_Y_sorted.ndim - 1:
        expanded_boundary_point = ref_point.unsqueeze(-2)
    else:
        view_shape = torch.Size([1] * len(batch_shape)) + torch.Size([1, 2])
        expanded_shape = batch_shape + torch.Size([1, 2])
        expanded_boundary_point = ref_point.view(view_shape).expand(expanded_shape)

    # add the points (ref, y) and (x, ref) to the corresponding ends
    left_end = torch.cat(
        [expanded_boundary_point[..., 0:1, 0:1], pareto_Y_sorted[..., 0:1, 1:2]], dim=-1
    )
    right_end = torch.cat(
        [pareto_Y_sorted[..., -1:, 0:1], expanded_boundary_point[..., 0:1, 1:2]], dim=-1
    )
    front = torch.cat([left_end, pareto_Y_sorted, right_end], dim=-2)
    # compute hypervolume by summing rectangles from min_x -> max_x
    top_lefts = torch.cat([front[..., :-1, 0:1], front[..., 1:, 1:2]], dim=-1)
    bottom_rights = torch.cat(
        [
            front[..., 1:, 0:1],
            expanded_boundary_point[..., 1:2].expand(*top_lefts.shape[:-1], 1),
        ],
        dim=-1,
    )
    # the last (unobserved) vertex in front is x= x_max, y = ref, which has an area of
    # zero. We could index the tensors the remove the last point, but that is likely
    # slower
    hv = (top_lefts - bottom_rights).abs().prod(dim=-1).sum(dim=-1)
    if compute_contributions:
        contributions = (
            (top_lefts[..., :-1, :] - top_lefts[..., 1:, :]).abs().prod(dim=-1)
        )
        # contributions is sorted, so we need to reverse the sorting
        sorting_n_shape = sorting_indices.shape[-1:]
        ordered_indices = (
            torch.arange(
                sorting_indices.shape[-1],
                dtype=torch.long,
                device=sorting_indices.device,
            )
            .view(torch.Size([1] * len(batch_shape)) + sorting_n_shape)
            .expand(batch_shape + sorting_n_shape)
        )
        inverse_sort = torch.empty_like(ordered_indices)
        inverse_sort.scatter_(dim=-1, index=sorting_indices, src=ordered_indices)
        contributions = contributions.gather(dim=-1, index=inverse_sort)
    else:
        contributions = None
    return hv, contributions


def get_reference_point(max_ref_point: Tensor, pareto_Y: Tensor) -> Tensor:
    r"""Set reference point.

    This sets the reference point to be `ref_point = nadir - 0.1 * nadir.abs()`
    when there is no pareto_Y that is better than the reference point.

    [Ishibuchi2011]_ find 0.1 to be a robust multiplier for scaling the
    nadir point.

    Note: this assumes maximization.

    Args:
        max_ref_point: a `m` dim tensor indicating the maximum reference point
        pareto_Y: a `n x m`-dim tensor of pareto points

    Returns:
        A `m`-dim tensor containing the reference point.
    """
    if (pareto_Y > max_ref_point).all(dim=-1).any():
        return max_ref_point
    nadir = pareto_Y.min(dim=0).values
    proposed_ref_point = nadir - 0.1 * nadir.abs()
    # make sure that the proposed_reference point is less than
    # or equal to max_ref_point
    return torch.min(proposed_ref_point, max_ref_point)