import torch
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}

from botorch.test_functions.multi_objective import BraninCurrin
from botorch.test_functions.multi_objective import VehicleSafety

# problem = BraninCurrin(negate=True)
# problem = BraninCurrin(negate=True).to(**tkwargs)
problem = VehicleSafety(negate=True).to(**tkwargs)

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples


def generate_initial_data(n=10):
    # generate training data
    train_x = draw_sobol_samples(bounds=problem.bounds, n=1, q=n, seed=torch.randint(600000, (1,)).item()).squeeze(0)
    train_obj = problem(train_x)
    print('train_x is', train_x)
    print('train_obj is', train_obj)
    return train_x, train_obj


def initialize_model(train_x, train_obj):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import time

import warnings

BATCH_SIZE = 1
# standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
standard_bounds[0] = 1
standard_bounds[1] = 3

def optimize_qehvi_and_get_observation(model, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    partitioning = NondominatedPartitioning(num_outcomes=problem.num_objectives, Y=train_obj)
    # print('this is our ref_point', problem.ref_point.tolist())
    print('ref point is', problem.ref_point.tolist())
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),  # use known reference point
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=1,
        raw_samples=2,  # used for intialization heuristic
        options={"batch_limit": 3, "maxiter": 200, "nonnegative": True},
        sequential=True,
    )
    # observe new values
    print('problem bound is', problem.bounds)
    print('old_x is', candidates)
    new_x = candidates.detach()
    # new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    print('new_x is', new_x)
    print('over')
    new_obj = problem(new_x)
    return new_x, new_obj



def optimize_qparego_and_get_observation(model, train_obj, sampler):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization
    of the qParEGO acquisition function, and returns a new candidate and observation."""
    acq_func_list = []
    for _ in range(BATCH_SIZE):
        weights = sample_simplex(problem.num_objectives, **tkwargs).squeeze()
        # weights = sample_simplex(problem.num_objectives).squeeze()
        objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=train_obj))
        acq_func = qExpectedImprovement(  # pyre-ignore: [28]
            model=model,
            objective=objective,
            best_f=objective(train_obj).max(),
            sampler=sampler,
        )
        acq_func_list.append(acq_func)
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=standard_bounds,
        num_restarts=5,
        raw_samples=256,  # used for intialization heuristic
        options={"batch_limit": 1, "maxiter": 2},
    )
    # observe new values
    new_x = unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj = problem(new_x)
    return new_x, new_obj


warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

N_TRIALS = 3
N_BATCH = 300
MC_SAMPLES = 128

verbose = True

hvs_qparego_all, hvs_qehvi_all, hvs_random_all = [], [], []

hv = Hypervolume(ref_point=problem.ref_point)
print('ref_point is located at', problem.ref_point)

# average over multiple trials

torch.manual_seed(1)


hvs_qparego, hvs_qehvi, hvs_random = [], [], []

# call helper functions to generate initial training data and initialize model
train_x_qparego, train_obj_qparego = generate_initial_data(n=10)
mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)

train_x_qehvi, train_obj_qehvi = train_x_qparego, train_obj_qparego
train_x_random, train_obj_random = train_x_qparego, train_obj_qparego
# compute hypervolume
mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)


# compute pareto front
pareto_mask = is_non_dominated(train_obj_qparego)
print('train_obj_qparego is', train_obj_qparego)
pareto_y = train_obj_qparego[pareto_mask]
print('pareto_y is', pareto_y)
# compute hypervolume

volume = hv.compute(pareto_y)
print('current hv is', volume)

# hvs_qparego.append(volume)
# hvs_qehvi.append(volume)
# hvs_random.append(volume)
#
#
# # run N_BATCH rounds of BayesOpt after the initial random batch
# for iteration in range(1, N_BATCH + 1):
#
#     t0 = time.time()
#
#     # fit the model
#     # fit_gpytorch_model(mll_qparego)
#     fit_gpytorch_model(mll_qehvi)
#
#     # define the qEI and qNEI acquisition modules using a QMC sampler
#
#     # qparego_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
#     qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
#
#     # optimize acquisition functions and get new observations
#
#     # new_x_qparego, new_obj_qparego = optimize_qparego_and_get_observation(
#     #     model_qparego, train_obj_qparego, qparego_sampler
#     # )
#     new_x_qehvi, new_obj_qehvi = optimize_qehvi_and_get_observation(
#         model_qehvi, train_obj_qehvi, qehvi_sampler
#     )
#     # new_x_random, new_obj_random = generate_initial_data(n=BATCH_SIZE)
#
#     # update training points
#     # train_x_qparego = torch.cat([train_x_qparego, new_x_qparego])
#     # train_obj_qparego = torch.cat([train_obj_qparego, new_obj_qparego])
#
#     train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
#     train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
#
#     # train_x_random = torch.cat([train_x_random, new_x_random])
#     # train_obj_random = torch.cat([train_obj_random, new_obj_random])
#
#     print('finish!!!!!!!')
#     # update progress
#     print('current random samples number is', len(train_obj_random))
#     print('current qparego samples number is', len(train_obj_qparego))
#     print('current qehvi samples number is', len(train_obj_qehvi))
#     for hvs_list, train_obj in zip(
#             (hvs_random, hvs_qparego, hvs_qehvi),
#             (train_obj_random, train_obj_qparego, train_obj_qehvi),
#     ):
#         # compute pareto front
#         pareto_mask = is_non_dominated(train_obj)
#         pareto_y = train_obj[pareto_mask]
#         # compute hypervolume
#         volume = hv.compute(pareto_y)
#         hvs_list.append(volume)
#
#     # reinitialize the models so they are ready for fitting on next iteration
#     # Note: we find improved performance from not warm starting the model hyperparameters
#     # using the hyperparameters from the previous iteration
#     # mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)
#     # print('current total sample number is', len(train_x_qparego))
#     mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
#
#     t1 = time.time()
#
#     if verbose:
#         print(
#             f"\nBatch {iteration:>2}: Hypervolume (random, qParEGO, qEHVI) = "
#             f"({hvs_random[-1]:>4.2f}, {hvs_qparego[-1]:>4.2f}, {hvs_qehvi[-1]:>4.2f}), "
#             f"time = {t1 - t0:>4.2f}.", end=""
#         )
#     else:
#         print(".", end="")
#     print('\n')
#     print('dfsfddas,', hvs_qehvi)
#     if iteration == N_BATCH:
#         with open('./res.txt', 'a') as f:
#             f.write(str(hvs_qparego))
#             f.write('\n')
#
#     pareto_mask = is_non_dominated(train_obj_qparego)
#     pareto_y = train_obj_qparego[pareto_mask]
#
#     saved_pareto = pareto_y.cpu().numpy().tolist()
#
#     with open('./samples/' + 'qparego_' + str(iteration) + '.txt', 'w') as f:
#         f.write(str(saved_pareto))
#
#
#
#
# # import matplotlib.pyplot as plt
# # import numpy as np
# # from Hypervolume import get_pareto
# #
# # pareto_mask = is_non_dominated(train_obj_qparego)
# # pareto_y = train_obj_qparego[pareto_mask]
# # print('pareto_y is', pareto_y)
# #
# #
# #
# # fig, ax = plt.subplots(figsize=(7, 5))
# # pareto_fX = pareto_y.cpu().numpy()
# # plt.scatter(pareto_fX[:, 0], pareto_fX[:, 1])
# # plt.xlim(-250, 0)
# # plt.ylim(-15, 0)
# #
# # plt.xlabel("pareto_Fx1")
# # plt.ylabel("pareto_Fx2")
# # fig.savefig('final_pareto_fronter.pdf', bbox_inches='tight')
#
#
#
#
# hvs_qparego_all.append(hvs_qparego)
# hvs_qehvi_all.append(hvs_qehvi)
# hvs_random_all.append(hvs_random)
#
# print(hvs_qehvi_all)
