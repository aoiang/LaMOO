import torch
import json
import numpy as np
from scipy.stats import norm
import copy as cp
from sklearn.svm import SVC
from sklearn.svm import SVR
from torch.quasirandom import SobolEngine
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from Hypervolume import get_pareto, compute_hypervolume_2d
import matplotlib.pyplot as plt
import random
from matplotlib import cm
import torch
from utils import convert_dtype

# from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
# from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
# from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
# from botorch.sampling.samplers import SobolQMCNormalSampler
# from botorch.utils.transforms import unnormalize

from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
import copy

from botorch.test_functions.multi_objective import VehicleSafety, BraninCurrin, DTLZ2
from moo_molecule_funcs.properties_obj3 import MOOMoleculeFunction
from moo_molecule_funcs.properties_obj3 import SUPPORTED_PROPERTIES

import pygmo as pg
from copy import deepcopy

from botorch_lamcts.botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch_lamcts.botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch_lamcts.botorch.sampling.samplers import SobolQMCNormalSampler
from botorch_lamcts.botorch.utils.transforms import unnormalize
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.acquisition.monte_carlo import qExpectedImprovement
# from MCTS import args


tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}



# the input will be samples!
class Classifier():
    def __init__(self, args, samples, dims):
        self.training_counter = 0
        assert dims >= 1
        assert type(samples) == type([])
        self.dims = dims

        #create a gaussian process regressor
        self.args = args
        noise = 0.1
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2) #default to CPU
        #learning boundary
        self.svm = SVC(kernel=args.kernel, gamma=args.gamma, degree=args.degree) # gamma is stable at scale
   
        #splitting samples
        self.regressor = SVR(kernel='linear', gamma='scale')
        self.label_id = None

        #data structures to store
        self.samples = np.array([])
        

        self.func = MOOMoleculeFunction(list(SUPPORTED_PROPERTIES.keys()))

        

     
        #good region is labeled as zero
        #bad  region is labeled as one
        self.good_label_mean = -1
        self.bad_label_mean = -1

        self.update_samples(samples)

    def update_samples(self, latest_samples):
        assert type(latest_samples) == type([])
        self.samples = latest_samples

    def get_hypervolume(self, ref_point):
        #the x_bar is the hyper-volume of current samples

        # X = torch.tensor(self.samples[:, 0])
        # Y = torch.tensor(self.samples[:, 1])

        X = self.samples[0]
        Y = self.samples[1]

        # ref_point = torch.tensor(ref_point)
        ref_point = ref_point.clone().detach()

        botorch_hv = Hypervolume(ref_point=ref_point)

        pareto_mask = is_non_dominated(Y)
        pareto_y = Y[pareto_mask]
        # print('is gpu ===========??????????????', pareto_y)
        hv = botorch_hv.compute(pareto_y)

        return hv

    def learn_boundary(self, plabel):
        assert len(plabel) == len(self.samples[0])
        #fitting a boundary in search space
        #plabel is from objective space
        #plabel_ss is from the search space
        plabel = plabel.ravel()
        X = self.samples[0]
        if torch.cuda.is_available():
            X = X.cpu().data.numpy()
        else:
            X = X.data.numpy()
        self.svm.fit(X, plabel)

    def viz_learned_boundary(self, node_name):
        X = np.random.rand(1000, 2)
        # X[:, 0] = X[:, 0] * 5
        # X[:, 1] = X[:, 1] * 3
        # f = BinhKorn()

        X[:, 0] = X[:, 0]
        X[:, 1] = X[:, 1]
        f = Branin_Currin()



        fX = np.array([f(x) for x in X])

        plabel_ss = self.svm.predict(X).reshape(-1)
        print("===>", fX.shape, plabel_ss.shape)
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.scatter(fX[np.where(plabel_ss==1)][:, 0], fX[np.where(plabel_ss==1)][:, 1])
        plt.scatter(fX[np.where(plabel_ss==0)][:, 0], fX[np.where(plabel_ss==0)][:, 1])
        plt.xlabel("f1")
        plt.ylabel("f2")
        fig.savefig(node_name + "_split_samples_in_stat.pdf", bbox_inches='tight')

        fig, ax = plt.subplots(figsize=(7, 5))
        plt.scatter(X[np.where(plabel_ss==1)][:, 0], X[np.where(plabel_ss==1)][:, 1])
        plt.scatter(X[np.where(plabel_ss==0)][:, 0], X[np.where(plabel_ss==0)][:, 1])
        plt.xlabel("x1")
        plt.ylabel("x2")
        fig.savefig(node_name + "_split_samples_in_stat_space.pdf", bbox_inches='tight')

    def viz_sample_clusters(self, node_name):
        #>>>>>>>>>>>>>drawing--------------
        fX = self.samples[:, 1]
        obj1 = fX[:, 0].reshape(-1, 1)
        obj2 = fX[:, 1].reshape(-1)
        plabel = self.learn_clusters()

        x = np.linspace(-300.0, 0.0, num=400)
        y = self.regressor.predict(x.reshape(-1, 1)).reshape(-1)
        obj1 = obj1.reshape(-1)
        obj2 = obj2.reshape(-1)
        fig, ax = plt.subplots(figsize=(7, 5))
        # print("scatter1:", obj1[np.where(plabel==1)], obj2[np.where(plabel==1)], type(obj1[np.where(plabel==1)]))
        plt.scatter(obj1[np.where(plabel==1)], obj2[np.where(plabel==1)])
        plt.scatter(obj1[np.where(plabel==0)], obj2[np.where(plabel==0)])
        plt.plot(x, y)
        plt.xlabel("f1")
        plt.ylabel("f2")
        fig.savefig(node_name+"_split_samples_in_obj.pdf", bbox_inches='tight')
        #>>>>>>>>>>>>>drawing--------------

    def learn_clusters(self, test_splittable=False):
        assert len(self.samples) >= 2, "samples must > 0"
        x = self.samples[0]
        fX = self.samples[1]
        #fitting a regressor between two objectives
        maximize = True

        pareto_mask = is_non_dominated(fX)
        pareto_fX = fX[pareto_mask]

        if torch.cuda.is_available():
            pareto_fX = pareto_fX.cpu().data.numpy()
            fX = fX.cpu().data.numpy()
        else:
            pareto_fX = pareto_fX.data.numpy()
            fX = fX.data.numpy()

        if test_splittable:
            self.label_id = random.randint(0, fX.shape[1]-1)
        # self.label_id = 0
        # label_id = 1
        print('label_id is', self.label_id)
        # label_id = 1
        obj_data_1 = fX[:, :self.label_id].reshape(fX.shape[0], -1)
        obj_data_2 = fX[:, self.label_id+1:].reshape(fX.shape[0], -1)
        # print('obj_data_1 is', obj_data_1, type(obj_data_1))
        # print('obj_data_2 is', obj_data_2, type(obj_data_2))
        obj_data = np.concatenate((obj_data_1, obj_data_2), axis=1)
        obj_label = fX[:, self.label_id].reshape(-1, 1)
        # print('obj_label is', obj_label)

        # print('obj_data test is', obj_data, type(obj_data))


        # print('fX here is', fX)

        # obj_data = fX[:, :fX.shape[1]-1].reshape(fX.shape[0], -1)
        # obj_label = fX[:, fX.shape[1]-1].reshape(-1, 1)

        # print('obj_data here is', obj_data)

        # obj_data = fX[:, 1:].reshape(fX.shape[0], -1)
        # obj_label = fX[:, 0].reshape(-1, 1)
        # print('obj_label is', obj_label)

        # obj_data = fX[:, 1].reshape(-1, 1)
        # obj_label = fX[:, 0].reshape(-1)

        copy_data = copy.deepcopy(obj_data)
        copy_label = copy.deepcopy(obj_label)

        # for i in range(len(obj_data[0])):
        #     min = copy_data[:, i].min()
        #     max = copy_data[:, i].max()
        #     copy_data[:, i] = (copy_data[:, i] + max) / -min

        # for i in range(len(obj_label)):
        min = copy_label.min()
        max = copy_label.max()

        # print('copy_label is:', copy_label)


        # print('normalized obj_data is', obj_data)
        try:
            self.regressor.fit(copy_data, obj_label)
        except:
            print(copy_data)
            print(obj_data)
            raise ValueError
        # self.regressor.fit(copy_data, copy_label)
        # reg_pred = self.regressor.predict(copy_data)

        # self.regressor.fit(obj_data, obj_label)
        # reg_pred = self.regressor.predict(obj_data)
        reg_pred = self.regressor.predict(copy_data)
        plabel = obj_label.reshape(-1) > reg_pred.reshape(-1) if maximize else obj_label.reshape(-1) < reg_pred.reshape(
            -1)

        # plabel = copy_label.reshape(-1) > reg_pred.reshape(-1) if maximize else copy_label.reshape(-1) < reg_pred.reshape(
        #     -1)

        # print('plabel is', plabel)



        #rectify the pareto front
        pareto_fX_check = np.array([item in pareto_fX for item in fX])
        # print('pareto_fX_check is', pareto_fX_check)
        plabel[np.where(pareto_fX_check == True)] = True
        # print('after plabel is', plabel)
        #change the label from boolean to 0~1
        plabel = plabel.astype(float)
        plabel = 1 - plabel# here 0 default to the good group, and 1 default to the bad group

        return plabel

    def hv_learn_clusters(self):
        ### this version is for HV
        assert len(self.samples) >= 2, "samples must > 0"
        x = self.samples[0]
        fX = self.samples[1]
        botorch_hv = Hypervolume(ref_point=self.func.ref_point.clone().detach())

        hvX = []
        for obj in fX:
            hvX.append(botorch_hv.compute(obj.reshape(-1, self.func.num_objectives)))

        sorted_hvX = copy.deepcopy(hvX)
        sorted_hvX.sort()
        # print('hvX is', hvX)
        avg_hv = sorted_hvX[len(hvX) - len(hvX) // 2]
        # print('avg hv is', avg_hv)
        plabel = []
        for p in hvX:
            plabel.append(p > avg_hv)
        plabel = np.array(plabel)
        plabel = plabel.astype(float)
        plabel = 1 - plabel
        # print('plabel here is:', plabel)
        pareto_mask = is_non_dominated(fX)
        pareto_fX = fX[pareto_mask]
        if torch.cuda.is_available():
            pareto_fX = pareto_fX.cpu().data.numpy()
            fX = fX.cpu().data.numpy()
        else:
            pareto_fX = pareto_fX.data.numpy()
            fX = fX.data.numpy()
        pareto_fX_check = np.array([item in pareto_fX for item in fX])
        # print('pareto_fX_check is', pareto_fX_check)
        plabel[np.where(pareto_fX_check == True)] = 0
        return plabel

    def dominance_learn_clusters(self):
        ### this version is for HV
        from copy import deepcopy
        assert len(self.samples) >= 2, "samples must > 0"
        x = self.samples[0]
        fX = self.samples[1]
        # botorch_hv = Hypervolume(ref_point=self.func.ref_point)

        obj_list = deepcopy(fX.cpu().numpy())
        obj_list *= -1

        ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=obj_list)
        # print(dc)

        id_dict = {}
        for i in range(len(dc)):
            id_dict[i] = dc[i]

        sorted_domi = np.array(sorted(id_dict.items(), key=lambda kv: (kv[1])))[:, 0]

        plabel = [1] * len(sorted_domi)

        for i in range(len(sorted_domi)):
            if i <= (len(sorted_domi) // 2):
                plabel[int(sorted_domi[i])] = 0

        plabel = np.array(plabel)
        return plabel




    def is_splittable_svm(self):
        if self.args.split_method == 'regressor':
            plabel = self.learn_clusters(True)
        else:
            plabel = self.dominance_learn_clusters()
        # print('plabel here is!!', plabel)
        if len(np.unique(plabel)) == 1:
            print('i am not splittable', plabel)
            return False
        self.learn_boundary(plabel)
        # x = self.samples[:, 0]
        if torch.cuda.is_available():
            x = self.samples[0].cpu().data.numpy()
        else:
            x = self.samples[0].data.numpy()
        svm_label = self.svm.predict(x)
        if len(np.unique(svm_label)) == 1:
            print('i am not splittable by svm_label', svm_label)
            return False
        else:
            return True


    def split_data(self):
        good_samples = []
        bad_samples = []
        if len(self.samples[0]) == 0:
            return good_samples, bad_samples

        if self.args.split_method == 'regressor':
            plabel = self.learn_clusters()
        else:
            plabel = self.dominance_learn_clusters()
        self.learn_boundary(plabel)
        assert len(plabel) == len(self.samples[0])

        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                if len(good_samples) == 0:
                    data = self.samples[0][idx].reshape(1, -1)
                    obj = self.samples[1][idx].reshape(1, -1)
                    good_samples.append(data)
                    good_samples.append(obj)
                else:
                    good_samples[0] = torch.cat([good_samples[0], self.samples[0][idx].reshape(1, -1)])
                    good_samples[1] = torch.cat([good_samples[1], self.samples[1][idx].reshape(1, -1)])
            else:
                if len(bad_samples) == 0:
                    data = self.samples[0][idx].reshape(1, -1)
                    obj = self.samples[1][idx].reshape(1, -1)
                    bad_samples.append(data)
                    bad_samples.append(obj)
                else:
                    bad_samples[0] = torch.cat([bad_samples[0], self.samples[0][idx].reshape(1, -1)])
                    bad_samples[1] = torch.cat([bad_samples[1], self.samples[1][idx].reshape(1, -1)])

        # print('bad sample is!', bad_samples[0])
        # print('length here is!', len(good_samples[0]), len(bad_samples[0]), len(self.samples[0]))


        assert len(good_samples[0]) + len(bad_samples[0]) == len(self.samples[0])
        return good_samples, bad_samples

    #############################################
    # visualize random sampling inside selected partition #
    #############################################

    def viz_node_sample_region(self, nums_samples, path, lb, ub, iter):
        from utils import latin_hypercube, from_unit_cube

        selected_cands = np.zeros((1, self.dims))
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(dimension=self.dims, scramble=True, seed=seed)

        # scale the samples to the entire search space
        # ----------------------------------- #

        while len(selected_cands) <= nums_samples:

            # cands = sobol.draw(200000).to(dtype=torch.float64).cpu().detach().numpy()
            # cands = (ub - lb) * cands + lb

            cands = latin_hypercube(1200000, self.dims)
            cands = from_unit_cube(cands, lb, ub)


            for node in path:
                boundary = node[0].classifier.svm
                if len(cands) == 0:
                    break
                    # return []
                cands = cands[boundary.predict(cands) == node[1]]  # node[1] store the direction to go
                if len(selected_cands) == 1:
                    print('cur_node is', node[0].get_name(), len(cands))
            selected_cands = np.append(selected_cands, cands, axis=0)

            print("total sampled:", len(selected_cands))

        final_cands_idx = np.random.choice(len(selected_cands), nums_samples)

        ########VIZ the samping region########
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.scatter(selected_cands[final_cands_idx][:, 0], selected_cands[final_cands_idx][:, 1])
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel("x1")
        plt.ylabel("x2")

        name = "leaf_node_samples_region" + str(iter) + '.png'

        fig.savefig(name, bbox_inches='tight')


        f = Branin_Currin()

        obj = []
        for sample in selected_cands[final_cands_idx]:
            value = f(sample)
            obj.append(np.array([sample, value]))

        obj = np.copy(obj)
        fX = obj[:, 1]

        fig, ax = plt.subplots(figsize=(7, 5))
        plt.scatter(fX[:, 0], fX[:, 1])
        plt.xlim([-300.0, 0.0])
        plt.ylim([-16.0, 0.0])
        plt.xlabel("f1")
        plt.ylabel("f2")

        name = "leaf_node_obj_region_" + str(iter) + '.png'

        fig.savefig(name, bbox_inches='tight')





    #############################################
    # random sampling inside selected partition #
    #############################################

    def propose_rand_samples_sobol(self, nums_samples, path, lb, ub, problem):
        #rejected sampling

        from utils import latin_hypercube, from_unit_cube
        from botorch.utils.sampling import draw_sobol_samples


        selected_cands = np.zeros((1, self.dims))
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(dimension=self.dims, scramble=True, seed=seed)



        bounds = torch.tensor([lb, ub], dtype=torch.float64)


        # scale the samples to the entire search space
        # ----------------------------------- #
        invalid_sample_times = 0
        # path = path[:1]
        while len(selected_cands) <= nums_samples:

            cands = latin_hypercube(15000, self.dims)
            cands = from_unit_cube(cands, lb, ub)

            for node in path:
                boundary = node[0].classifier.svm
                if len(cands) == 0:
                    break
                cands = cands[boundary.predict(cands) == node[1]] # node[1] store the direction to go
                if len(selected_cands) == 1:
                    print('cur_node is', node[0].get_name(), len(cands))

            selected_cands = np.append(selected_cands, cands, axis=0)
            print("total sampled:", len(selected_cands))
            path = path[:-1]

        selected_cands = selected_cands[1:]
        final_cands_idx = np.random.choice(len(selected_cands), nums_samples)

        if not torch.cuda.is_available():
            new_x = torch.tensor(selected_cands[final_cands_idx])
        else:
            new_x = torch.tensor(selected_cands[final_cands_idx], device='cuda')
        new_obj = problem(new_x)

        return new_x, new_obj

    #############################################
    # random sampling inside selected partition #
    #############################################

    def propose_rand_samples_cmaes(self, num_samples, path, lb, ub, func, init_within_leaf, LEAF_SAMPLE_SIZE=5,
                                   vanilla=False, samples=None):

        if vanilla:
            self.samples = samples

        def lamcts_raw_samples(path, num_samples, es):
            new_X = []
            times = 0
            p = 0
            while len(new_X) <= num_samples:
                # print('num_samples are', num_samples)
                cands = es.ask()
                cands = np.array(cands)
                for node in path:
                    boundary = node[0].classifier.svm
                    if len(cands) == 0:
                        break
                    # print('current cands are', cands)
                    # print(boundary.predict(cands))
                    cands = cands[boundary.predict(cands) == node[1]]  # node[1] store the direction to go
                    # if len(new_X) == 1:
                    #     print('cur_node is', node[0].get_name(), len(cands))

                # new_X = np.append(new_X, cands, axis=0)
                new_X.extend(cands)
                print('cur path is', p)
                print("total sampled_cmaes:", len(new_X))
                times += 1
                #if times >= 5:
                path = path[:-1]
                #    times = 0
                #    p += 1
            new_X = new_X[:num_samples]
            return new_X


        import cma
        import contextlib

        # print('len self.X', len(self.X))
        pareto_fX = is_non_dominated(self.samples[1]).cpu().numpy()





        pareto_indices = np.where(pareto_fX == True)[0]
        # print('pareto_indices are', pareto_indices)
        # print('pareto_fX is', pareto_fX)

        sample_X = self.samples[0].cpu().numpy()
        sample_fX = self.samples[1].cpu().numpy()

        x = self.samples[0]
        fX = self.samples[1]



        botorch_hv = Hypervolume(ref_point=self.func.ref_point.clone().detach())


        hvX = []
        for obj in fX:
            hvX.append(botorch_hv.compute(obj.reshape(-1, self.func.num_objectives)))

        # print('botorch_hv is', hvX)
        max_hvX = max(hvX)
        for index in pareto_indices:
            hvX[index] = max_hvX + random.random()


        # print("sample_X is", sample_X)
        # print("sample_fX is", sample_fX)
        if len(sample_X) > num_samples:  # since we're adding more samples as we go, start from the best few
            best_indices = sorted(list(range(len(sample_X))), key=lambda i: hvX[i], reverse=True)
            # print('best_indices is', best_indices)
            tell_X, tell_fX = np.stack(
                [sample_X[i] for i in best_indices[:max(LEAF_SAMPLE_SIZE, num_samples)]], axis=0), np.stack(
                [sample_fX[i] for i in best_indices[:max(LEAF_SAMPLE_SIZE, num_samples)]], axis=0)
        else:
            tell_X, tell_fX = sample_X, np.array([fx for fx in sample_fX])
        print('=================tell_fX is==============\n', tell_fX)
        print(tell_fX.argmax())
        if init_within_leaf == 'mean':
            x0 = np.mean(tell_X, axis=0)
        elif init_within_leaf == 'random':
            x0 = random.choice(tell_X)
        elif init_within_leaf == 'max':
            x0 = tell_X[tell_fX.argmax()]  # start from the best
        else:
            raise NotImplementedError
        sigma0 = np.mean([np.std(tell_X[:, i]) for i in range(tell_X.shape[1])])
        sigma0 = max(1, sigma0)  # clamp min 1
        # sigma0 = min(1, sigma0)
        sigma0 *= 0.15
        # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        es = cma.CMAEvolutionStrategy(x0, sigma0, {'maxfevals': num_samples, 'popsize': max(2, len(tell_X)),
                                                   'bounds': [lb, ub], 'seed': 86})
        num_evals = 0
        proposed_X, fX, split_info, aux_info = [], [], [], []
        if vanilla:
            init_X = es.ask()
        else:
            init_X = lamcts_raw_samples(path, num_samples, es)
        if len(tell_X) < 2:
            pad_X = init_X[:2 - len(tell_X)]
            pad_fX = [func(x) for x in pad_X]
            proposed_X += pad_X

            fX += [tup[0] for tup in pad_fX]
            # split_info += [tup[1] for tup in pad_fX]
            # aux_info += [tup[2] for tup in pad_fX]
            es.tell([x for x in tell_X] + pad_X, [fx for fx in tell_fX] + [tup[0] for tup in pad_fX])
            num_evals += 2 - len(tell_X)
        else:
            # print('fadsfdasfadsfdasfdas')
            es.tell(tell_X, tell_fX)
        while num_evals < num_samples:
            if vanilla:
                new_X = es.ask()
            else:
                new_X = lamcts_raw_samples(path, num_samples, es)






            # print('cxxkfjdsakfjaksjf is', new_X)
            # print('new_X is', new_X)
            # print('xxxx is', es.ask())
            if num_evals + len(new_X) > num_samples:
                random.shuffle(new_X)
                new_X = new_X[:num_samples - num_evals]
                # new_fX = [func(x) for x in new_X]
                try:
                    new_fX = func(torch.tensor(new_X))
                except:
                    continue
                new_fX = new_fX.cpu().numpy()
            else:
                # new_fX = [func(x) for x in new_X]
                try:
                    new_fX = func(torch.tensor(new_X))
                except:
                    continue
                new_fX = new_fX.cpu().numpy()
                # print('new_fX is', new_fX)
                es.tell(new_X, [tup[0] for tup in new_fX])
            proposed_X += new_X
            fX += [tup[0] for tup in new_fX]
            # split_info += [tup[1] for tup in new_fX]
            # aux_info += [tup[2] for tup in new_fX]
            num_evals += len(new_fX)
        assert num_evals == num_samples
        if torch.cuda.is_available():
            proposed_X = torch.tensor(proposed_X, device='cuda')
        else:
            proposed_X = torch.tensor(proposed_X)
        new_obj = func(proposed_X)
        # print('proposed_X is', proposed_X)
        return proposed_X, new_obj

    def propose_rand_samples_cmaes2(self, num_samples, path, lb, ub, func, init_within_leaf, LEAF_SAMPLE_SIZE=5,
                                   vanilla=False, samples=None):

        if vanilla:
            self.samples = samples

        def lamcts_raw_samples(path, num_samples, es):
            new_X = []
            times = 0
            p = 0
            while len(new_X) <= num_samples:
                # print('num_samples are', num_samples)
                cands = es.ask()
                cands = np.array(cands)
                for node in path:
                    boundary = node[0].classifier.svm
                    if len(cands) == 0:
                        break
                    # print('current cands are', cands)
                    # print(boundary.predict(cands))
                    cands = cands[boundary.predict(cands) == node[1]]  # node[1] store the direction to go
                    # if len(new_X) == 1:
                    #     print('cur_node is', node[0].get_name(), len(cands))

                # new_X = np.append(new_X, cands, axis=0)
                new_X.extend(cands)
                print('cur path is', p)
                print("total sampled_cmaes:", len(new_X))
                times += 1
                if times >= 5:
                    path = path[:-1]
                    times = 0
                    p += 1
            new_X = new_X[:num_samples]
            return new_X

        def cal_do(fx):

            cur_samples = self.samples[1]
            l = len(fx)
            fx = torch.tensor(fx, device='cuda')
            cur_samples = torch.cat([cur_samples, fx])
            obj_list = deepcopy(cur_samples.cpu().numpy())
            obj_list *= -1
            ndf, dl, dc_pos, ndr = pg.fast_non_dominated_sorting(points=obj_list)
            # dc = [[val * 1] for val in dc_pos]

            dc = [[float(max(dc_pos) - val)] for val in dc_pos]
            print('cal_do dc is===========', dc)
            # print('hello world')
            # print(dc[-l:])
            # print(torch.tensor(dc[-l:], device='cuda'))


            return torch.tensor(dc[-l:], device='cuda')



        import cma
        import contextlib
        from copy import deepcopy
        # print('len self.X', len(self.X))
        # print('heihei', self.samples[1])
        pareto_fX = is_non_dominated(self.samples[1]).cpu().numpy()

        obj_list = deepcopy(self.samples[1].cpu().numpy())
        obj_list *= -1

        ndf, dl, dc_pos, ndr = pg.fast_non_dominated_sorting(points=obj_list)
        # dc = [val * 1 for val in dc_pos]

        dc = [[float(max(dc_pos) - val)] for val in dc_pos]
        print('===================dc==============\n', dc)


        # pareto_indices = np.where(pareto_fX == True)[0]
        # print('pareto_indices are', pareto_indices)
        # print('pareto_fX is', pareto_fX)

        sample_X = self.samples[0].cpu().numpy()
        sample_fX = self.samples[1].cpu().numpy()

        x = self.samples[0]
        fX = self.samples[1]

        botorch_hv = Hypervolume(ref_point=self.func.ref_point.clone().detach())

        # hvX = []
        # for obj in fX:
        #     hvX.append(botorch_hv.compute(obj.reshape(-1, self.func.num_objectives)))
        #
        # # print('botorch_hv is', hvX)
        # max_hvX = max(hvX)
        # for index in pareto_indices:
        #     hvX[index] = max_hvX + random.random()

        # print("sample_X is", sample_X)
        # print("sample_fX is", sample_fX)
        if len(sample_X) > num_samples:  # since we're adding more samples as we go, start from the best few
            best_indices = sorted(list(range(len(sample_X))), key=lambda i: dc[i], reverse=True)
            print('best_indices is==============', best_indices)
            tell_X, tell_fX = np.stack(
                [sample_X[i] for i in best_indices[:max(LEAF_SAMPLE_SIZE, num_samples)]], axis=0), np.stack(
                [dc[i] for i in best_indices[:max(LEAF_SAMPLE_SIZE, num_samples)]], axis=0)
        else:
            tell_X, tell_fX = sample_X, np.array([fx for fx in sample_fX])
        print('=================tell_fX is==============\n', tell_fX)
        print(tell_fX.argmax())
        if init_within_leaf == 'mean':
            x0 = np.mean(tell_X, axis=0)
        elif init_within_leaf == 'random':
            x0 = random.choice(tell_X)
        elif init_within_leaf == 'max':
            x0 = tell_X[tell_fX.argmax()]  # start from the best
        else:
            raise NotImplementedError
        sigma0 = np.mean([np.std(tell_X[:, i]) for i in range(tell_X.shape[1])])
        sigma0 = max(1, sigma0)  # clamp min 1
        # sigma0 = min(1, sigma0)
        sigma0 *= 0.99
        # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        es = cma.CMAEvolutionStrategy(x0, sigma0, {'maxfevals': num_samples, 'popsize': max(2, len(tell_X)),
                                                   'bounds': [lb, ub], 'seed': 86})
        num_evals = 0
        proposed_X, fX, split_info, aux_info = [], [], [], []
        if vanilla:
            init_X = es.ask()
        else:
            init_X = lamcts_raw_samples(path, num_samples, es)
        if len(tell_X) < 2:
            pad_X = init_X[:2 - len(tell_X)]
            # pad_fX = [cal_hv(x) for x in pad_X]
            pad_funcX = [func(x) for x in pad_X]
            pad_fX = [cal_do(x) for x in pad_X]



            proposed_X += pad_X

            fX += [tup[0] for tup in pad_fX]
            # split_info += [tup[1] for tup in pad_fX]
            # aux_info += [tup[2] for tup in pad_fX]
            es.tell([x for x in tell_X] + pad_X, [fx for fx in tell_fX] + [tup[0] for tup in pad_fX])
            num_evals += 2 - len(tell_X)
        else:
            # print('fadsfdasfadsfdasfdas')
            es.tell(tell_X, tell_fX)
        while num_evals < num_samples:
            if vanilla:
                new_X = es.ask()
            else:
                new_X = lamcts_raw_samples(path, num_samples, es)

            # print('cxxkfjdsakfjaksjf is', new_X)
            # print('new_X is', new_X)
            # print('xxxx is', es.ask())
            if num_evals + len(new_X) > num_samples:
                random.shuffle(new_X)
                new_X = new_X[:num_samples - num_evals]
                # new_fX = [func(x) for x in new_X]
                try:
                    new_fX = cal_do(torch.tensor(func(new_X)))
                except:
                    continue

                new_fX = new_fX.cpu().numpy()
            else:
                # new_fX = [func(x) for x in new_X]
                try:
                    new_fX = cal_do(torch.tensor(func(new_X)))
                except:
                    continue
                new_fX = new_fX.cpu().numpy()

                # print('new_fX is', new_fX)
                es.tell(new_X, [tup[0] for tup in new_fX])
            proposed_X += new_X
            fX += [tup[0] for tup in new_fX]
            # split_info += [tup[1] for tup in new_fX]
            # aux_info += [tup[2] for tup in new_fX]
            num_evals += len(new_fX)
        assert num_evals == num_samples
        if torch.cuda.is_available():
            proposed_X = torch.tensor(proposed_X, device='cuda')
        else:
            proposed_X = torch.tensor(proposed_X)
        new_obj = func(proposed_X)
        # print('proposed_X is', proposed_X)
        return proposed_X, new_obj




    #############################################
    # Bayesian Optimization sampling inside selected partition #
    #############################################

    def propose_rand_samples_bayesian(self, nums_samples, path, lb, ub, model, mll, train_obj, func, acq_func='qehvi', cur_iter=0):

        BATCH_SIZE = nums_samples
        if not torch.cuda.is_available():
            standard_bounds = torch.tensor([lb, ub], dtype=torch.float64)
        else:
            standard_bounds = torch.tensor([lb, ub], dtype=torch.float64, device='cuda')

            # print('type of standard bounds is', type(standard_bounds))

        sampler = SobolQMCNormalSampler(num_samples=128)
        # print('I AM THE BBBBBBBBBBBBBBBBBBOUNDRY IN classifier', path)

        def optimize_qehvi_and_get_observation(model, train_obj, sampler, path):
            """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
            # partition non-dominated space into disjoint rectangles
            # partitioning = NondominatedPartitioning(num_outcomes=func.num_objectives, Y=train_obj)
            partitioning = NondominatedPartitioning(ref_point=func.ref_point, Y=train_obj)
            acq_func = qExpectedHypervolumeImprovement(
                model=model,
                ref_point=func.ref_point.tolist(),  # use known reference point
                partitioning=partitioning,
                sampler=sampler,
            )
            # optimize
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=standard_bounds,
                q=BATCH_SIZE,
                num_restarts=20,
                raw_samples=256,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 48, "nonnegative": True},
                sequential=True,
                problem=func,
                lamcts_boundry=path,
                cur_iter=cur_iter,
            )

            new_x = candidates.detach()

            new_obj = func(new_x)
            return new_x, new_obj


        def optimize_qparego_and_get_observation(model, train_obj, sampler, path):
            """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization
            of the qParEGO acquisition function, and returns a new candidate and observation."""
            acq_func_list = []
            for _ in range(BATCH_SIZE):
                weights = sample_simplex(2, **tkwargs).squeeze()
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
                lamcts_boundry=path,
            )
            # observe new values

            new_x = unnormalize(candidates.detach(), bounds=torch.tensor([(0.0, 0.0), (1.0, 1.0)], dtype=torch.float))
            return new_x.cpu().detach().numpy()

            # new_obj = problem(new_x)
            # return new_x, new_obj

        new_samples = None
        new_obj = None

        if acq_func == 'qehvi':
            new_samples, new_obj = optimize_qehvi_and_get_observation(model, train_obj, sampler, path)
        elif acq_func == 'qparego':
            new_samples = optimize_qparego_and_get_observation(model, train_obj, sampler, path)


        return new_samples, new_obj

