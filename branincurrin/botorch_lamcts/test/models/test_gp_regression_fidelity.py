#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import warnings

import torch
from botorch import fit_gpytorch_model
from botorch.exceptions.errors import UnsupportedError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.gp_regression_fidelity import (
    FixedNoiseMultiFidelityGP,
    SingleTaskMultiFidelityGP,
)
from botorch.models.transforms import Normalize, Standardize
from botorch.posteriors import GPyTorchPosterior
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.containers import TrainingData
from botorch.utils.testing import BotorchTestCase, _get_random_data
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood


def _get_random_data_with_fidelity(batch_shape, m, n_fidelity, n=10, **tkwargs):
    r"""Construct test data.
    For this test, by convention the trailing dimesions are the fidelity dimensions
    """
    train_x, train_y = _get_random_data(batch_shape, m, n, **tkwargs)
    s = torch.rand(n, n_fidelity, **tkwargs).repeat(batch_shape + torch.Size([1, 1]))
    train_x = torch.cat((train_x, s), dim=-1)
    train_y = train_y + (1 - s).pow(2).sum(dim=-1).unsqueeze(-1)
    return train_x, train_y


class TestSingleTaskMultiFidelityGP(BotorchTestCase):

    FIDELITY_TEST_PAIRS = ((None, 1), (1, None), (None, -1), (-1, None), (1, 2))

    def _get_model_and_data(
        self,
        iteration_fidelity,
        data_fidelity,
        batch_shape,
        m,
        lin_truncated,
        outcome_transform=None,
        input_transform=None,
        **tkwargs,
    ):
        n_fidelity = (iteration_fidelity is not None) + (data_fidelity is not None)
        train_X, train_Y = _get_random_data_with_fidelity(
            batch_shape=batch_shape, m=m, n_fidelity=n_fidelity, **tkwargs
        )
        model_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "iteration_fidelity": iteration_fidelity,
            "data_fidelity": data_fidelity,
            "linear_truncated": lin_truncated,
        }
        if outcome_transform is not None:
            model_kwargs["outcome_transform"] = outcome_transform
        if input_transform is not None:
            model_kwargs["input_transform"] = input_transform
        model = SingleTaskMultiFidelityGP(**model_kwargs)
        return model, model_kwargs

    def test_init_error(self):
        train_X = torch.rand(2, 2, device=self.device)
        train_Y = torch.rand(2, 1)
        for lin_truncated in (True, False):
            with self.assertRaises(UnsupportedError):
                SingleTaskMultiFidelityGP(
                    train_X, train_Y, linear_truncated=lin_truncated
                )

    def test_gp(self):
        for (iteration_fidelity, data_fidelity) in self.FIDELITY_TEST_PAIRS:
            num_dim = 1 + (iteration_fidelity is not None) + (data_fidelity is not None)
            bounds = torch.zeros(2, num_dim)
            bounds[1] = 1
            for (
                batch_shape,
                m,
                dtype,
                lin_trunc,
                use_octf,
                use_intf,
            ) in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
                (False, True),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                octf = Standardize(m=m, batch_shape=batch_shape) if use_octf else None
                intf = Normalize(d=num_dim, bounds=bounds) if use_intf else None
                model, model_kwargs = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelity=data_fidelity,
                    batch_shape=batch_shape,
                    m=m,
                    lin_truncated=lin_trunc,
                    outcome_transform=octf,
                    input_transform=intf,
                    **tkwargs,
                )
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                mll.to(**tkwargs)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=OptimizationWarning)
                    fit_gpytorch_model(mll, sequential=False, options={"maxiter": 1})

                # test init
                self.assertIsInstance(model.mean_module, ConstantMean)
                self.assertIsInstance(model.covar_module, ScaleKernel)
                if use_octf:
                    self.assertIsInstance(model.outcome_transform, Standardize)
                if use_intf:
                    self.assertIsInstance(model.input_transform, Normalize)
                    # permute output dim
                    train_X, train_Y, _ = model._transform_tensor_args(
                        X=model_kwargs["train_X"], Y=model_kwargs["train_Y"]
                    )
                    # check that the train inputs have been transformed and set on the
                    # model
                    self.assertTrue(torch.equal(model.train_inputs[0], intf(train_X)))

                # test param sizes
                params = dict(model.named_parameters())
                for p in params:
                    self.assertEqual(
                        params[p].numel(), m * torch.tensor(batch_shape).prod().item()
                    )

                # test posterior
                # test non batch evaluation
                X = torch.rand(*batch_shape, 3, num_dim, **tkwargs)
                expected_shape = batch_shape + torch.Size([3, m])
                posterior = model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(posterior.mean.shape, expected_shape)
                self.assertEqual(posterior.variance.shape, expected_shape)
                if use_octf:
                    # ensure un-transformation is applied
                    tmp_tf = model.outcome_transform
                    del model.outcome_transform
                    pp_tf = model.posterior(X)
                    model.outcome_transform = tmp_tf
                    expected_var = tmp_tf.untransform_posterior(pp_tf).variance
                    self.assertTrue(torch.allclose(posterior.variance, expected_var))

                # test batch evaluation
                X = torch.rand(2, *batch_shape, 3, num_dim, **tkwargs)
                expected_shape = torch.Size([2]) + batch_shape + torch.Size([3, m])
                posterior = model.posterior(X)
                self.assertIsInstance(posterior, GPyTorchPosterior)
                self.assertEqual(posterior.mean.shape, expected_shape)
                self.assertEqual(posterior.variance.shape, expected_shape)
                if use_octf:
                    # ensure un-transformation is applied
                    tmp_tf = model.outcome_transform
                    del model.outcome_transform
                    pp_tf = model.posterior(X)
                    model.outcome_transform = tmp_tf
                    expected_var = tmp_tf.untransform_posterior(pp_tf).variance
                    self.assertTrue(torch.allclose(posterior.variance, expected_var))

    def test_condition_on_observations(self):
        for (iteration_fidelity, data_fidelity) in self.FIDELITY_TEST_PAIRS:
            n_fidelity = (iteration_fidelity is not None) + (data_fidelity is not None)
            num_dim = 1 + n_fidelity
            for batch_shape, m, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, model_kwargs = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelity=data_fidelity,
                    batch_shape=batch_shape,
                    m=m,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )
                # evaluate model
                model.posterior(torch.rand(torch.Size([4, num_dim]), **tkwargs))
                # test condition_on_observations
                fant_shape = torch.Size([2])
                # fantasize at different input points
                X_fant, Y_fant = _get_random_data_with_fidelity(
                    fant_shape + batch_shape, m, n_fidelity=n_fidelity, n=3, **tkwargs
                )
                c_kwargs = (
                    {"noise": torch.full_like(Y_fant, 0.01)}
                    if isinstance(model, FixedNoiseGP)
                    else {}
                )
                cm = model.condition_on_observations(X_fant, Y_fant, **c_kwargs)
                # fantasize at different same input points
                c_kwargs_same_inputs = (
                    {"noise": torch.full_like(Y_fant[0], 0.01)}
                    if isinstance(model, FixedNoiseGP)
                    else {}
                )
                cm_same_inputs = model.condition_on_observations(
                    X_fant[0], Y_fant, **c_kwargs_same_inputs
                )

                test_Xs = [
                    # test broadcasting single input across fantasy and
                    # model batches
                    torch.rand(4, num_dim, **tkwargs),
                    # separate input for each model batch and broadcast across
                    # fantasy batches
                    torch.rand(batch_shape + torch.Size([4, num_dim]), **tkwargs),
                    # separate input for each model and fantasy batch
                    torch.rand(
                        fant_shape + batch_shape + torch.Size([4, num_dim]), **tkwargs
                    ),
                ]
                for test_X in test_Xs:
                    posterior = cm.posterior(test_X)
                    self.assertEqual(
                        posterior.mean.shape,
                        fant_shape + batch_shape + torch.Size([4, m]),
                    )
                    posterior_same_inputs = cm_same_inputs.posterior(test_X)
                    self.assertEqual(
                        posterior_same_inputs.mean.shape,
                        fant_shape + batch_shape + torch.Size([4, m]),
                    )

                    # check that fantasies of batched model are correct
                    if len(batch_shape) > 0 and test_X.dim() == 2:
                        state_dict_non_batch = {
                            key: (val[0] if val.numel() > 1 else val)
                            for key, val in model.state_dict().items()
                        }

                        model_kwargs_non_batch = {}
                        for k, v in model_kwargs.items():
                            if k in (
                                "iteration_fidelity",
                                "data_fidelity",
                                "linear_truncated",
                                "input_transform",
                            ):
                                model_kwargs_non_batch[k] = v
                            else:
                                model_kwargs_non_batch[k] = v[0]

                        model_non_batch = type(model)(**model_kwargs_non_batch)
                        model_non_batch.load_state_dict(state_dict_non_batch)
                        model_non_batch.eval()
                        model_non_batch.likelihood.eval()
                        model_non_batch.posterior(
                            torch.rand(torch.Size([4, num_dim]), **tkwargs)
                        )
                        c_kwargs = (
                            {"noise": torch.full_like(Y_fant[0, 0, :], 0.01)}
                            if isinstance(model, FixedNoiseGP)
                            else {}
                        )
                        mnb = model_non_batch
                        cm_non_batch = mnb.condition_on_observations(
                            X_fant[0][0], Y_fant[:, 0, :], **c_kwargs
                        )
                        non_batch_posterior = cm_non_batch.posterior(test_X)
                        self.assertTrue(
                            torch.allclose(
                                posterior_same_inputs.mean[:, 0, ...],
                                non_batch_posterior.mean,
                                atol=1e-3,
                            )
                        )
                        self.assertTrue(
                            torch.allclose(
                                posterior_same_inputs.mvn.covariance_matrix[:, 0, :, :],
                                non_batch_posterior.mvn.covariance_matrix,
                                atol=1e-3,
                            )
                        )

    def test_fantasize(self):
        for (iteration_fidelity, data_fidelity) in self.FIDELITY_TEST_PAIRS:
            n_fidelity = (iteration_fidelity is not None) + (data_fidelity is not None)
            num_dim = 1 + n_fidelity
            for batch_shape, m, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, model_kwargs = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelity=data_fidelity,
                    batch_shape=batch_shape,
                    m=m,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )
                # fantasize
                X_f = torch.rand(
                    torch.Size(batch_shape + torch.Size([4, num_dim])), **tkwargs
                )
                sampler = SobolQMCNormalSampler(num_samples=3)
                fm = model.fantasize(X=X_f, sampler=sampler)
                self.assertIsInstance(fm, model.__class__)
                fm = model.fantasize(X=X_f, sampler=sampler, observation_noise=False)
                self.assertIsInstance(fm, model.__class__)

    def test_subset_model(self):
        for (iteration_fidelity, data_fidelity) in self.FIDELITY_TEST_PAIRS:
            num_dim = 1 + (iteration_fidelity is not None) + (data_fidelity is not None)
            for batch_shape, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, _ = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelity=data_fidelity,
                    batch_shape=batch_shape,
                    m=2,
                    lin_truncated=lin_trunc,
                    outcome_transform=None,  # TODO: Subset w/ outcome transform
                    **tkwargs,
                )
                subset_model = model.subset_output([0])
                X = torch.rand(
                    torch.Size(batch_shape + torch.Size([3, num_dim])), **tkwargs
                )
                p = model.posterior(X)
                p_sub = subset_model.posterior(X)
                self.assertTrue(
                    torch.allclose(p_sub.mean, p.mean[..., [0]], atol=1e-4, rtol=1e-4)
                )
                self.assertTrue(
                    torch.allclose(
                        p_sub.variance, p.variance[..., [0]], atol=1e-4, rtol=1e-4
                    )
                )

    def test_construct_inputs(self):
        for (iteration_fidelity, data_fidelity) in self.FIDELITY_TEST_PAIRS:
            for batch_shape, m, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, model_kwargs = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelity=data_fidelity,
                    batch_shape=batch_shape,
                    m=m,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )
                # len(Xs) == len(Ys) == 1
                training_data = TrainingData(
                    X=model_kwargs["train_X"],
                    Y=model_kwargs["train_Y"],
                    Yvar=torch.full_like(model_kwargs["train_Y"], 0.01),
                )
                # missing fidelity features
                with self.assertRaises(ValueError):
                    model.construct_inputs(training_data)
                data_dict = model.construct_inputs(training_data, fidelity_features=[1])
                self.assertTrue("train_Yvar" not in data_dict)
                self.assertTrue("data_fidelity" in data_dict)
                self.assertEqual(data_dict["data_fidelity"], 1)
                data_dict = model.construct_inputs(training_data, fidelity_features=[1])
                self.assertTrue(
                    torch.equal(data_dict["train_X"], model_kwargs["train_X"])
                )
                self.assertTrue(
                    torch.equal(data_dict["train_Y"], model_kwargs["train_Y"])
                )


class TestFixedNoiseMultiFidelityGP(TestSingleTaskMultiFidelityGP):
    def _get_model_and_data(
        self,
        iteration_fidelity,
        data_fidelity,
        batch_shape,
        m,
        lin_truncated,
        outcome_transform=None,
        input_transform=None,
        **tkwargs,
    ):
        n_fidelity = (iteration_fidelity is not None) + (data_fidelity is not None)
        train_X, train_Y = _get_random_data_with_fidelity(
            batch_shape=batch_shape, m=m, n_fidelity=n_fidelity, **tkwargs
        )
        train_Yvar = torch.full_like(train_Y, 0.01)
        model_kwargs = {
            "train_X": train_X,
            "train_Y": train_Y,
            "train_Yvar": train_Yvar,
            "iteration_fidelity": iteration_fidelity,
            "data_fidelity": data_fidelity,
            "linear_truncated": lin_truncated,
        }
        if outcome_transform is not None:
            model_kwargs["outcome_transform"] = outcome_transform
        if input_transform is not None:
            model_kwargs["input_transform"] = input_transform
        model = FixedNoiseMultiFidelityGP(**model_kwargs)
        return model, model_kwargs

    def test_init_error(self):
        train_X = torch.rand(2, 2, device=self.device)
        train_Y = torch.rand(2, 1)
        train_Yvar = torch.full_like(train_Y, 0.01)
        for lin_truncated in (True, False):
            with self.assertRaises(UnsupportedError):
                FixedNoiseMultiFidelityGP(
                    train_X, train_Y, train_Yvar, linear_truncated=lin_truncated
                )

    def test_fixed_noise_likelihood(self):
        for (iteration_fidelity, data_fidelity) in self.FIDELITY_TEST_PAIRS:
            for batch_shape, m, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, model_kwargs = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelity=data_fidelity,
                    batch_shape=batch_shape,
                    m=m,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )
                self.assertIsInstance(model.likelihood, FixedNoiseGaussianLikelihood)
                self.assertTrue(
                    torch.equal(
                        model.likelihood.noise.contiguous().view(-1),
                        model_kwargs["train_Yvar"].contiguous().view(-1),
                    )
                )

    def test_construct_inputs(self):
        for (iteration_fidelity, data_fidelity) in self.FIDELITY_TEST_PAIRS:
            for batch_shape, m, dtype, lin_trunc in itertools.product(
                (torch.Size(), torch.Size([2])),
                (1, 2),
                (torch.float, torch.double),
                (False, True),
            ):
                tkwargs = {"device": self.device, "dtype": dtype}
                model, model_kwargs = self._get_model_and_data(
                    iteration_fidelity=iteration_fidelity,
                    data_fidelity=data_fidelity,
                    batch_shape=batch_shape,
                    m=m,
                    lin_truncated=lin_trunc,
                    **tkwargs,
                )
                training_data = TrainingData(
                    X=model_kwargs["train_X"], Y=model_kwargs["train_Y"]
                )
                # missing Yvars
                with self.assertRaises(ValueError):
                    model.construct_inputs(training_data, fidelity_features=[1])
                # len(Xs) == len(Ys) == 1
                training_data = TrainingData(
                    X=model_kwargs["train_X"],
                    Y=model_kwargs["train_Y"],
                    Yvar=torch.full_like(model_kwargs["train_Y"], 0.01),
                )
                # missing fidelity features
                with self.assertRaises(ValueError):
                    model.construct_inputs(training_data)
                data_dict = model.construct_inputs(training_data, fidelity_features=[1])
                self.assertTrue("train_Yvar" in data_dict)
                self.assertTrue("data_fidelity" in data_dict)
                self.assertEqual(data_dict["data_fidelity"], 1)
                data_dict = model.construct_inputs(training_data, fidelity_features=[1])
                self.assertTrue(
                    torch.equal(data_dict["train_X"], model_kwargs["train_X"])
                )
                self.assertTrue(
                    torch.equal(data_dict["train_Y"], model_kwargs["train_Y"])
                )
