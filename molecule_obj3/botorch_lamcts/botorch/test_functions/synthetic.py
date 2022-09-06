#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Synthetic functions for optimization benchmarks.
Reference: https://www.sfu.ca/~ssurjano/optimization.html
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from botorch.test_functions.base import BaseTestProblem
from torch import Tensor


class SyntheticTestFunction(BaseTestProblem):
    r"""Base class for synthetic test functions."""

    _optimizers: List[Tuple[float, ...]]
    _optimal_value: float
    num_objectives: int = 1

    def __init__(self, noise_std: Optional[float] = None, negate: bool = False) -> None:
        r"""Base constructor for synthetic test functions.

        Args:
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
        """
        super().__init__(noise_std=noise_std, negate=negate)
        if self._optimizers is not None:
            self.register_buffer(
                "optimizers", torch.tensor(self._optimizers, dtype=torch.float)
            )

    @property
    def optimal_value(self) -> float:
        r"""The global minimum (maximum if negate=True) of the function."""
        return -self._optimal_value if self.negate else self._optimal_value


class Ackley(SyntheticTestFunction):
    r"""Ackley test function.

    d-dimensional function (usually evaluated on `[-32.768, 32.768]^d`):

        f(x) = -A exp(-B sqrt(1/d sum_{i=1}^d x_i^2)) -
            exp(1/d sum_{i=1}^d cos(c x_i)) + A + exp(1)

    f has one minimizer for its global minimum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0`.
    """

    _optimal_value = 0.0
    _check_grad_at_opt: bool = False

    def __init__(
        self, dim: int = 2, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self._bounds = [(-32.768, 32.768) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)
        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi

    def evaluate_true(self, X: Tensor) -> Tensor:
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b / math.sqrt(self.dim) * torch.norm(X, dim=-1))
        part2 = -(torch.exp(torch.mean(torch.cos(c * X), dim=-1)))
        return part1 + part2 + a + math.e


class Beale(SyntheticTestFunction):

    dim = 2
    _optimal_value = 0.0
    _bounds = [(-4.5, 4.5), (-4.5, 4.5)]
    _optimizers = [(3.0, 0.5)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        part1 = (1.5 - x1 + x1 * x2) ** 2
        part2 = (2.25 - x1 + x1 * x2 ** 2) ** 2
        part3 = (2.625 - x1 + x1 * x2 ** 3) ** 2
        return part1 + part2 + part3


class Branin(SyntheticTestFunction):
    r"""Branin test function.

    Two-dimensional function (usually evaluated on `[-5, 10] x [0, 15]`):

        B(x) = (x_2 - b x_1^2 + c x_1 - r)^2 + 10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`
    B has 3 minimizers for its global minimum at `z_1 = (-pi, 12.275)`,
    `z_2 = (pi, 2.275)`, `z_3 = (9.42478, 2.475)` with `B(z_i) = 0.397887`.
    """

    dim = 2
    _bounds = [(-5.0, 10.0), (0.0, 15.0)]
    _optimal_value = 0.397887
    _optimizers = [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        t1 = (
            X[..., 1]
            - 5.1 / (4 * math.pi ** 2) * X[..., 0] ** 2
            + 5 / math.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
        return t1 ** 2 + t2 + 10


class Bukin(SyntheticTestFunction):

    dim = 2
    _bounds = [(-15.0, -5.0), (-3.0, 3.0)]
    _optimal_value = 0.0
    _optimizers = [(-10.0, 1.0)]
    _check_grad_at_opt: bool = False

    def evaluate_true(self, X: Tensor) -> Tensor:
        part1 = 100.0 * torch.sqrt(torch.abs(X[..., 1] - 0.01 * X[..., 0] ** 2))
        part2 = 0.01 * torch.abs(X[..., 0] + 10.0)
        return part1 + part2


class Cosine8(SyntheticTestFunction):
    r"""Cosine Mixture test function.

    8-dimensional function (usually evaluated on `[-1, 1]^8`):

        f(x) = 0.1 sum_{i=1}^8 cos(5 pi x_i) - sum_{i=1}^8 x_i^2

    f has one maximizer for its global maximum at `z_1 = (0, 0, ..., 0)` with
    `f(z_1) = 0.8`
    """

    dim = 8
    _bounds = [(-1.0, 1.0) for _ in range(8)]
    _optimal_value = 0.8
    _optimizers = [tuple(0.0 for _ in range(8))]

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.sum(0.1 * torch.cos(5 * math.pi * X) - X ** 2, dim=-1)


class DropWave(SyntheticTestFunction):

    dim = 2
    _bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    _optimal_value = -1.0
    _optimizers = [(0.0, 0.0)]
    _check_grad_at_opt = False

    def evaluate_true(self, X: Tensor) -> Tensor:
        norm = torch.norm(X, dim=-1)
        part1 = 1.0 + torch.cos(12.0 * norm)
        part2 = 0.5 * norm.pow(2) + 2.0
        return -part1 / part2


class DixonPrice(SyntheticTestFunction):

    _optimal_value = 0.0

    def __init__(
        self, dim=2, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self._bounds = [(-10.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [
            tuple(
                math.pow(2.0, -(1.0 - 2.0 ** (-(i - 1))))
                for i in range(1, self.dim + 1)
            )
        ]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        d = self.dim
        part1 = (X[..., 0] - 1) ** 2
        i = X.new(range(2, d + 1))
        part2 = torch.sum(i * (2.0 * X[..., 1:] ** 2 - X[..., :-1]) ** 2, dim=1)
        return part1 + part2


class EggHolder(SyntheticTestFunction):
    r"""Eggholder test function.

    Two-dimensional function (usually evaluated on `[-512, 512]^2`):

        E(x) = (x_2 + 47) sin(R1(x)) - x_1 * sin(R2(x))

    where `R1(x) = sqrt(|x_2 + x_1 / 2 + 47|)`, `R2(x) = sqrt|x_1 - (x_2 + 47)|)`.
    """

    dim = 2
    _bounds = [(-512.0, 512.0), (-512.0, 512.0)]
    _optimal_value = -959.6407
    _optimizers = [(512.0, 404.2319)]
    _check_grad_at_opt: bool = False

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        part1 = -(x2 + 47.0) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2.0 + 47.0)))
        part2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47.0))))
        return part1 + part2


class Griewank(SyntheticTestFunction):

    _optimal_value = 0.0

    def __init__(
        self, dim=2, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self._bounds = [(-600.0, 600.0) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        part1 = torch.sum(X ** 2 / 4000.0, dim=1)
        d = X.shape[1]
        part2 = -(
            torch.prod(torch.cos(X / torch.sqrt(X.new(range(1, d + 1))).view(1, -1)))
        )
        return part1 + part2 + 1.0


class Hartmann(SyntheticTestFunction):
    r"""Hartmann synthetic test function.

    Most commonly used is the six-dimensional version (typically evaluated on
    `[0, 1]^6`):

        H(x) = - sum_{i=1}^4 ALPHA_i exp( - sum_{j=1}^6 A_ij (x_j - P_ij)**2 )

    H has a 6 local minima and a global minimum at

        z = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)

    with `H(z) = -3.32237`.
    """

    def __init__(
        self, dim=6, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        if dim not in (3, 4, 6):
            raise ValueError(f"Hartmann with dim {dim} not defined")
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        # optimizers and optimal values for dim=4 not implemented
        optvals = {3: -3.86278, 6: -3.32237}
        optimizers = {
            3: [(0.114614, 0.555649, 0.852547)],
            6: [(0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)],
        }
        self._optimal_value = optvals.get(self.dim)
        self._optimizers = optimizers.get(self.dim)
        super().__init__(noise_std=noise_std, negate=negate)
        self.register_buffer("ALPHA", torch.tensor([1.0, 1.2, 3.0, 3.2]))
        if dim == 3:
            A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            P = [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        elif dim == 4:
            A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            P = [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        elif dim == 6:
            A = [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
            P = [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        self.register_buffer("A", torch.tensor(A, dtype=torch.float))
        self.register_buffer("P", torch.tensor(P, dtype=torch.float))

    @property
    def optimal_value(self) -> float:
        if self.dim == 4:
            raise NotImplementedError()
        return super().optimal_value

    @property
    def optimizers(self) -> Tensor:
        if self.dim == 4:
            raise NotImplementedError()
        return super().optimizers

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        inner_sum = torch.sum(self.A * (X.unsqueeze(1) - 0.0001 * self.P) ** 2, dim=2)
        H = -(torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=1))
        if self.dim == 4:
            H = (1.1 + H) / 0.839
        return H


class HolderTable(SyntheticTestFunction):
    r"""Holder Table synthetic test function.

    Two-dimensional function (typically evaluated on `[0, 10] x [0, 10]`):

        `H(x) = - | sin(x_1) * cos(x_2) * exp(| 1 - ||x|| / pi | ) |`

    H has 4 global minima with `H(z_i) = -19.2085` at

        z_1 = ( 8.05502,  9.66459)
        z_2 = (-8.05502, -9.66459)
        z_3 = (-8.05502,  9.66459)
        z_4 = ( 8.05502, -9.66459)
    """

    dim = 2
    _bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    _optimal_value = -19.2085
    _optimizers = [
        (8.05502, 9.66459),
        (-8.05502, -9.66459),
        (-8.05502, 9.66459),
        (8.05502, -9.66459),
    ]

    def evaluate_true(self, X: Tensor) -> Tensor:
        term = torch.abs(1 - torch.norm(X, dim=1) / math.pi)
        return -(
            torch.abs(torch.sin(X[..., 0]) * torch.cos(X[..., 1]) * torch.exp(term))
        )


class Levy(SyntheticTestFunction):
    r"""Levy synthetic test function.

    d-dimensional function (usually evaluated on `[-10, 10]^d`):

        f(x) = sin^2(pi w_1) +
            sum_{i=1}^{d-1} (w_i-1)^2 (1 + 10 sin^2(pi w_i + 1)) +
            (w_d - 1)^2 (1 + sin^2(2 pi w_d))

    where `w_i = 1 + (x_i - 1) / 4` for all `i`.

    f has one minimizer for its global minimum at `z_1 = (1, 1, ..., 1)` with
    `f(z_1) = 0`.
    """

    _optimal_value = 0.0

    def __init__(
        self, dim=2, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self._bounds = [(-10.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        w = 1.0 + (X - 1.0) / 4.0
        part1 = torch.sin(math.pi * w[..., 0]) ** 2
        part2 = torch.sum(
            (w[..., :-1] - 1.0) ** 2
            * (1.0 + 10.0 * torch.sin(math.pi * w[..., :-1] + 1.0) ** 2),
            dim=1,
        )
        part3 = (w[..., -1] - 1.0) ** 2 * (
            1.0 + torch.sin(2.0 * math.pi * w[..., -1]) ** 2
        )
        return part1 + part2 + part3


class Michalewicz(SyntheticTestFunction):
    r"""Michalewicz synthetic test function.

    d-dim function (usually evaluated on hypercube [0, pi]^d):

        M(x) = sum_{i=1}^d sin(x_i) (sin(i x_i^2 / pi)^20)
    """

    def __init__(
        self, dim=2, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self._bounds = [(0.0, math.pi) for _ in range(self.dim)]
        optvals = {2: -1.80130341, 5: -4.687658, 10: -9.66015}
        optimizers = {2: [(2.20290552, 1.57079633)]}
        self._optimal_value = optvals.get(self.dim)
        self._optimizers = optimizers.get(self.dim)
        super().__init__(noise_std=noise_std, negate=negate)
        self.register_buffer(
            "i", torch.tensor(tuple(range(1, self.dim + 1)), dtype=torch.float)
        )

    @property
    def optimizers(self) -> Tensor:
        if self.dim in (5, 10):
            raise NotImplementedError()
        return super().optimizers

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        m = 10
        return -(
            torch.sum(
                torch.sin(X) * torch.sin(self.i * X ** 2 / math.pi) ** (2 * m), dim=-1
            )
        )


class Powell(SyntheticTestFunction):

    _optimal_value = 0.0

    def __init__(
        self, dim=4, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self._bounds = [(-4.0, 5.0) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        result = torch.zeros_like(X[..., 0])
        for i in range(self.dim // 4):
            i_ = i + 1
            part1 = (X[..., 4 * i_ - 4] + 10.0 * X[..., 4 * i_ - 3]) ** 2
            part2 = 5.0 * (X[..., 4 * i_ - 2] - X[..., 4 * i_ - 1]) ** 2
            part3 = (X[..., 4 * i_ - 3] - 2.0 * X[..., 4 * i_ - 2]) ** 4
            part4 = 10.0 * (X[..., 4 * i_ - 4] - X[..., 4 * i_ - 1]) ** 4
            result += part1 + part2 + part3 + part4
        return result


class Rastrigin(SyntheticTestFunction):

    _optimal_value = 0.0

    def __init__(
        self, dim=2, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self._bounds = [(-5.12, 5.12) for _ in range(self.dim)]
        self._optimizers = [tuple(0.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return 10.0 * self.dim + torch.sum(
            X ** 2 - 10.0 * torch.cos(2.0 * math.pi * X), dim=-1
        )


class Rosenbrock(SyntheticTestFunction):
    r"""Rosenbrock synthetic test function.

    d-dimensional function (usually evaluated on `[-5, 10]^d`):

        f(x) = sum_{i=1}^{d-1} (100 (x_{i+1} - x_i^2)^2 + (x_i - 1)^2)

    f has one minimizer for its global minimum at `z_1 = (1, 1, ..., 1)` with
    `f(z_i) = 0.0`.
    """

    _optimal_value = 0.0

    def __init__(
        self, dim=2, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self._bounds = [(-5.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(1.0 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.sum(
            100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (X[..., :-1] - 1) ** 2,
            dim=-1,
        )


class Shekel(SyntheticTestFunction):
    r"""Shekel synthtetic test function.

    4-dimensional function (usually evaluated on `[0, 10]^4`):

        f(x) = -sum_{i=1}^10 (sum_{j=1}^4 (x_j - A_{ji})^2 + C_i)^{-1}

    f has one minimizer for its global minimum at `z_1 = (4, 4, 4, 4)` with
    `f(z_1) = -10.5363`.
    """

    dim = 4
    _bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 10.0), (0.0, 10.0)]
    _optimizers = [(4.000747, 3.99951, 4.00075, 3.99951)]

    def __init__(
        self, m: int = 10, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.m = m
        optvals = {5: -10.1532, 7: -10.4029, 10: -10.536443}
        self._optimal_value = optvals[self.m]
        super().__init__(noise_std=noise_std, negate=negate)

        self.register_buffer(
            "beta", torch.tensor([1, 2, 2, 4, 4, 6, 3, 7, 5, 5], dtype=torch.float)
        )
        C_t = torch.tensor(
            [
                [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
                [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
                [4, 1, 8, 6, 7, 9, 3, 1, 2, 3.6],
            ],
            dtype=torch.float,
        )
        self.register_buffer("C", C_t.transpose(-1, -2))

    def evaluate_true(self, X: Tensor) -> Tensor:
        self.to(device=X.device, dtype=X.dtype)
        beta = self.beta / 10.0
        result = -sum(
            1 / (torch.sum((X - self.C[i]) ** 2, dim=-1) + beta[i])
            for i in range(self.m)
        )
        return result


class SixHumpCamel(SyntheticTestFunction):

    dim = 2
    _bounds = [(-3.0, 3.0), (-2.0, 2.0)]
    _optimal_value = -1.0316
    _optimizers = [(0.0898, -0.7126), (-0.0898, 0.7126)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        return (
            (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2
            + x1 * x2
            + (4 * x2 ** 2 - 4) * x2 ** 2
        )


class StyblinskiTang(SyntheticTestFunction):
    r"""Styblinski-Tang synthtetic test function.

    d-dimensional function (usually evaluated on the hypercube `[-5, 5]^d`):

        H(x) = 0.5 * sum_{i=1}^d (x_i^4 - 16 * x_i^2 + 5 * x_i)

    H has a single global mininimum `H(z) = -39.166166 * d` at `z = [-2.903534]^d`
    """

    def __init__(
        self, dim=2, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self._bounds = [(-5.0, 5.0) for _ in range(self.dim)]
        self._optimal_value = -39.166166 * self.dim
        self._optimizers = [tuple(-2.903534 for _ in range(self.dim))]
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return 0.5 * (X ** 4 - 16 * X ** 2 + 5 * X).sum(dim=1)


class ThreeHumpCamel(SyntheticTestFunction):

    dim = 2
    _bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    _optimal_value = 0.0
    _optimizers = [(0.0, 0.0)]

    def evaluate_true(self, X: Tensor) -> Tensor:
        x1, x2 = X[..., 0], X[..., 1]
        return 2.0 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 6.0 + x1 * x2 + x2 ** 2
