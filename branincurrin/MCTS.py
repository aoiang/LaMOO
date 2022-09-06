import json
import collections
import copy as cp
import math
from collections import OrderedDict
import os.path
import numpy as np
import time
import operator
import sys
import pickle
import os
import random
from datetime import datetime
from Node import Node
from utils import latin_hypercube, from_unit_cube, convert_dtype
import argparse
import copy



from torch.quasirandom import SobolEngine
import torch
import matplotlib.pyplot as plt
from botorch.test_functions.multi_objective import VehicleSafety, BraninCurrin, DTLZ2



from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.utils.sampling import draw_sobol_samples
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
from botorch.utils.sampling import draw_sobol_samples
# from functions.nasbench101 import Nasbench101

import warnings



class MCTS:
    #############################################

    def __init__(self, lb, ub, dims, ninits, func, args, run=0):
        self.dims = dims
        self.samples = []
        self.nodes = []
        self.Cp = args.cp
        self.lb = lb
        self.ub = ub
        self.ninits = ninits
        self.func = func
        self.best_value_trace = []
        self.sample_counter = 0
        self.visualization = False
        self.LEAF_SAMPLE_SIZE = 1
        self.run_times = run
        self.args = args
        self.discreted_sample_track = {}
        
        #we start the most basic form of the tree, 3 nodes and height = 1
        root = Node(args=self.args, parent=None, dims=self.dims, reset_id=True, cp=self.Cp)
        self.nodes.append(root)
        
        self.ROOT = root
        self.CURT = self.ROOT
        self.samples_hv_track = []
        self.random_samples = []

        self.plot_samples = latin_hypercube(100000, self.dims)
        self.plot_samples = from_unit_cube(self.plot_samples, self.lb, self.ub)

        self.init_train()
        self.init_surrogate_model()


    def init_surrogate_model(self):
        train_x = self.samples[0]
        train_obj = self.samples[1]


        self.surrogate_model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
        self.mll = ExactMarginalLogLikelihood(self.surrogate_model.likelihood, self.surrogate_model)
        fit_gpytorch_model(self.mll)

    def populate_training_data(self):
        #only keep root
        self.ROOT.obj_counter = 0
        for node in self.nodes:
            node.clear_data()
        self.nodes.clear()
        new_root = Node(args=self.args, parent=None, dims=self.dims, reset_id=True, cp=self.Cp)
        self.nodes.append(new_root)


        self.ROOT = new_root
        self.CURT = self.ROOT
        self.ROOT.update_bag(self.samples, self.func.ref_point)
    
    def get_leaf_status(self):
        status = []

        for node in self.nodes:
            if node.is_leaf() is True and len(node.bag[0]) > self.LEAF_SAMPLE_SIZE and node.is_svm_splittable is True:
                status.append(True)
            else:
                status.append(False)
        return np.array(status)
        
    def get_split_idx(self):
        split_by_samples = np.argwhere(self.get_leaf_status() == True).reshape(-1)
        return split_by_samples
    
    def is_splitable(self):
        status = self.get_leaf_status()
        if True in status:
            return True
        else:
            return False
        
    def dynamic_treeify(self):
        # we bifurcate a node once it contains over 20 samples
        # the node will bifurcate into a good and a bad kid
        self.populate_training_data()
        print("total nodes:", len(self.nodes))
        assert len(self.ROOT.bag[0]) == len(self.samples[0])
        assert len(self.nodes) == 1

        
        while self.is_splitable():
            to_split = self.get_split_idx()
            print("==>to split:", to_split, " total:", len(self.nodes))
            for nidx in to_split:
                parent = self.nodes[nidx] # parent check if the boundary is splittable by svm
                assert len(parent.bag[0]) >= self.LEAF_SAMPLE_SIZE
                assert parent.is_svm_splittable is True
                # print("spliting node:", parent.get_name(), len(parent.bag))
                good_kid_data, bad_kid_data = parent.train_and_split()
                #creat two kids, assign the data, and push into lists
                # children's lb and ub will be decided by its parent

                assert len(good_kid_data[0]) + len(bad_kid_data[0]) == len(parent.bag[0])
                assert len(good_kid_data[0]) > 0
                assert len(bad_kid_data[0]) > 0

                good_kid = Node(args=self.args, parent=parent, dims=self.dims, reset_id=False, cp=self.Cp)
                bad_kid = Node(args=self.args, parent=parent, dims=self.dims, reset_id=False, cp=self.Cp)


                good_kid.update_bag(good_kid_data, self.func.ref_point)
                bad_kid.update_bag(bad_kid_data, self.func.ref_point)

            
                parent.update_kids(good_kid=good_kid, bad_kid=bad_kid)

                if good_kid.get_xbar() == 0.0:
                    good_kid.x_bar += 0.1
            
                self.nodes.append(good_kid)
                self.nodes.append(bad_kid)
            if len(self.nodes) > 50:
                break
                
            print("continue split:", self.is_splitable())

            #CAUTION: make sure the good kid in into list first
        
        self.print_tree()
        # self.track_nodes()
        # self.viz_tree()


        
    def collect_samples(self, sample, value=None):
        if value == None:
            value = self.func(sample)

        sample = sample.tolist()
        value = value.tolist()
        self.sample_counter += 1
        self.samples.append(np.array([sample, value]))

        #each entry [0]: samples, [1]: vaues
        return value

    def init_train(self):

        if not torch.cuda.is_available():
            data = torch.load('BraninCurrin_data.pt', map_location=torch.device('cpu'))
        else:
            data = torch.load('BraninCurrin_data.pt')
        init_samples, init_obj = data[self.run_times][0], data[self.run_times][1]
        if torch.cuda.is_available():
            init_samples, init_obj = torch.tensor(init_samples, device='cuda'), torch.tensor(init_obj, device='cuda')
            # init_samples, init_obj = init_samples.clone(device='cuda').detach(), init_obj.clone(device='cuda').detach()


#        print('init_obj is', init_obj)

        self.samples.append(init_samples)
        self.samples.append(init_obj)

        self.random_samples.append(init_samples)
        self.random_samples.append(init_obj)
        
        print("="*10 + 'collect ' + str(len(self.samples[0])) +' points for initializing MCTS'+"="*10)
        print("lb:", self.lb)
        print("ub:", self.ub)
        print("Cp:", self.Cp)
        print("inits:", self.ninits)
        print("dims:", self.dims)
        print("="*58)
        
    def print_tree(self):
        print('-'*100)
        for node in self.nodes:
            print(node)
        print('-'*100)

    def track_nodes(self, kid='good'):
        assert len(self.nodes) > 0
        node = self.nodes[0]

        selected_nodes = []
        hvs = []

        while not node.is_leaf():
            selected_nodes.append(node.get_name())
            hvs.append(node.get_xbar())
            node = node.kids[0] if kid == 'good' else node.kids[1]

        plt.bar(x=selected_nodes, height=hvs, label='hv')
        plt.ylim(4.7, 5.3)
        plt.savefig('progress_chart.png')


    def reset_to_root(self):
        self.CURT = self.ROOT
    
    def load_agent(self):
        node_path = 'mcts_agent'
        if os.path.isfile(node_path) == True:
            with open(node_path, 'rb') as json_data:
                self = pickle.load(json_data)
                print("=====>loads:", len(self.samples)," samples" )

    def dump_agent(self):
        node_path = 'mcts_agent'
        print("dumping the agent.....")
        with open(node_path,"wb") as outfile:
            pickle.dump(self, outfile)
            
    def dump_samples(self):
        sample_path = 'samples_'+str(self.sample_counter)
        with open(sample_path, "wb") as outfile:
            pickle.dump(self.samples, outfile)
    
    def dump_trace(self):
        trace_path = 'results/result'+str(self.sample_counter)
        final_results_str = json.dumps(self.best_value_trace)
        with open(trace_path, "a") as f:
            f.write(final_results_str + '\n')

    def greedy_select(self):
        self.reset_to_root()
        print('this is the greedy select')
        curt_node = self.ROOT
        path = []
        if self.visualization == True:
            curt_node.plot_samples_and_boundary(self.func)

        try:
            curt_node.visualize_node()
        except:
            print('node samples are not svm splittable')

        while not curt_node.is_leaf():
            UCT = []
            k_good = {}
            k_good['good'] = -1
            k_good['bad'] = -1
            for i in curt_node.kids:
                if i.is_good_kid:
                    k_good['good'] = i.get_xbar()
                else:
                    k_good['bad'] = i.get_xbar()
                if k_good['good'] == k_good['bad'] and i.is_good_kid:
                    UCT.append(i.get_xbar() + 0.01)
                else:
                    UCT.append(i.get_xbar())
            # print(UCT)

            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]
            if not curt_node.is_leaf() and self.visualization:
                curt_node.plot_samples_and_boundary(self.func)
            try:
                curt_node.visualize_node()
            except:
                print('node samples are not svm splittable!')
            print("=>", curt_node.get_name(), choice,  end=' ')

        try:
            curt_node.visualize_node()
        except:
            print('node samples are not svm splittable')
        print("")
        return curt_node, path
        

    def select(self):
        self.reset_to_root()
        curt_node = self.ROOT

        path = []
        while not curt_node.is_leaf():
            UCT = []
            for i in curt_node.kids:
                UCT.append(i.get_uct())
            choice = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path.append((curt_node, choice))
            curt_node = curt_node.kids[choice]
            # try:
            #     curt_node.visualize_node()
            # except:
            #     print('node samples are not svm splittable')
            print("=>", curt_node.get_name(), choice, end=' ')
            print('\n')

        try:
            curt_node.visualize_node()
        except:
            print('node samples are not svm splittable')
        # print(path)
        # print(path[-1][0].get_name())
        return curt_node, path

    def leaf_select(self):
        self.reset_to_root()
        curt_node = self.ROOT
        leaves = []

        for node in self.nodes:
            if node.is_leaf():
                leaves.append(node)

        paths = []
        for leaf in leaves:
            path = []
            while leaf is not self.ROOT:
                path.insert(0, (leaf.parent, 0 if leaf.is_good_kid() else 1))
                leaf = leaf.parent
            paths.append(path)

        # print(paths)

        if len(paths) > 1:
            UCT = []
            for i in range(len(paths)):
                UCT.append(paths[i][-1][0].kids[paths[i][-1][1]].get_uct())
            print(UCT)

            leaf = np.random.choice(np.argwhere(UCT == np.amax(UCT)).reshape(-1), 1)[0]
            path = paths[leaf]
            curt_node = path[-1][0]
        else:
            path = paths[0]
            curt_node = self.ROOT

        dis = []
        for node in path:
            dis.append((node[0].get_name(), node[1]))

        try:
            curt_node.visualize_node()
        except:
            print('node samples are not svm splittable')
        # print(path)
        return curt_node, path
    
    def backpropogate(self, leaf, acc):
        curt_node = leaf
        while curt_node is not None:
            assert curt_node.n > 0
            curt_node.x_bar = (curt_node.x_bar*curt_node.n + acc) / (curt_node.n + 1)
            curt_node.n += 1
            curt_node = curt_node.parent


    def search(self):


        botorch_hv = Hypervolume(ref_point=self.func.ref_point.clone().detach())

        hv_track = []

        sample_num = self.args.sample_num
        print('MAX HV IS!!!!!!!!!', self.func._max_hv)
        time_list = []

        train_obj = self.samples[1]
        pareto_mask = is_non_dominated(train_obj)
        pareto_y = train_obj[pareto_mask]

        if torch.cuda.is_available():
            pareto_y = torch.tensor(pareto_y, device='cuda')

        hv = botorch_hv.compute(pareto_y)
        print('current iteration botorch hv is', hv)
        hv_track.append(hv)






        for i in range(0, self.args.iter):
            # print('cur samples nums are', self.samples)
            t0 = time.time()
            if self.args.cmaes_method == 'lamcts':
                self.dynamic_treeify()
            t2 = time.time()
            print(
                f"time = {t2 - t0:>4.2f}.", end=""
            )
            print('\n')
            if self.args.node_select == 'mcts':
                leaf, path = self.select()
            elif self.args.node_select == 'leaf':

                leaf, path = self.leaf_select()
            else:
                leaf, path = self.greedy_select()
            self.samples_hv_track.append(self.nodes[0].get_xbar())






            for j in range(0, 1):


                train_obj = self.samples[1]


                if self.args.sample_method == 'bayesian':

                    samples, objs = leaf.propose_samples_Bayesian_Optimization(sample_num, path, self.lb, self.ub, self.surrogate_model, self.mll, train_obj, self.func, cur_iter=i)

                elif self.args.sample_method == 'cmaes':
                    if self.args.cmaes_method == 'lamcts':
                        samples, objs = leaf.propose_samples_cmaes(sample_num, path, self.lb, self.ub,
                                                                   self.func, vanilla=False, samples=self.samples)
                    else:
                        samples, objs = leaf.propose_samples_cmaes(sample_num, path, self.lb, self.ub,
                                                                   self.func, vanilla=True, samples=self.samples)
                else:
                    samples, objs = leaf.propose_samples_sobol(sample_num, path, self.lb, self.ub, self.func)


                self.samples[0] = torch.cat([self.samples[0], samples])
                self.samples[1] = torch.cat([self.samples[1], objs])



                selected_cands = np.zeros((1, self.dims))
                seed = np.random.randint(int(1e6))

                cands = latin_hypercube(20000, self.dims)
                cands = from_unit_cube(cands, self.lb, self.ub)
                selected_cands = np.append(selected_cands, cands, axis=0)
                selected_cands = selected_cands[1:]
                final_cands_idx = np.random.choice(len(selected_cands), sample_num)











                ### botorch HV

                # print('000000=is =========', self.samples[0])

                train_obj = self.samples[1]
                pareto_mask = is_non_dominated(train_obj)
                pareto_y = train_obj[pareto_mask]
                # print('pareto_y is=====', pareto_y.data.numpy().tolist())
                # print('y is=====', train_obj.data.numpy().tolist())

                if torch.cuda.is_available():
                    pareto_y = torch.tensor(pareto_y, device='cuda')

                hv = botorch_hv.compute(pareto_y)

                hv_track.append(hv)
                print('hv in each iter is', hv_track)


            if self.args.sample_method == 'bayesian':
                self.init_surrogate_model()
            if len(self.samples[0]) > 10000:
                print("total samples:", len(self.samples[0]))
                break
            # print('samples hv each iter is', self.samples_hv_track)

            t1 = time.time()

            print(
                f"time = {t1 - t0:>4.2f}.", end=""
            )
            print('\n')

            time_list.append(t1 - t0)



        print('hv in each iter is', hv_track)





if __name__ == '__main__':
    parser = argparse.ArgumentParser("MCTS")
    parser.add_argument('--problem', type=str, default='bc', help='choose the problem')
    parser.add_argument('--data_id', type=int, default=-1, help='specific run id')
    parser.add_argument('--kernel', type=str, default='poly', help='kernel type of svm')
    parser.add_argument('--gamma', type=str, default='scale', help='auto or scale')
    parser.add_argument('--degree', type=int, default=4, help='svm degree')
    parser.add_argument('--iter', type=int, default=18, help='total iterations')
    parser.add_argument('--sample_num', type=int, default=5, help='sample numsbers per iteration')
    parser.add_argument('--runs', type=int, default=5, help='total runs')
    parser.add_argument('--cp', type=float, default=6, help='cp value in MCTS')
    parser.add_argument('--sample_method', type=str, default='bayesian', help='bayesian, cmaes or random')
    parser.add_argument('--cmaes_method', type=str, default='lamcts', help='lamcts or vanilla')
    parser.add_argument('--split_method', type=str, default='dominance', help='dominance or regressor')
    parser.add_argument('--node_select', type=str, default='leaf', help='mcts or leaf')

    args = parser.parse_args()

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # print('gpu is', torch.cuda.is_available())

    f = BraninCurrin(negate=True).to(**tkwargs)

    if args.data_id < 0:
        for i in range(args.runs):
            if not torch.cuda.is_available():
                agent = MCTS(lb=f.bounds[0].data.numpy(), ub=f.bounds[1].data.numpy(), dims=f.dim, ninits=10, func=f,
                             args=args, run=i)
            else:
                agent = MCTS(lb=f.bounds[0].cpu().data.numpy(), ub=f.bounds[1].cpu().data.numpy(), dims=f.dim,
                             ninits=10, func=f, args=args,
                             run=i)
            agent.search()
    else:
        if not torch.cuda.is_available():
            agent = MCTS(lb=f.bounds[0].data.numpy(), ub=f.bounds[1].data.numpy(), dims=f.dim, ninits=10, func=f,
                         args=args, run=args.data_id)
        else:
            agent = MCTS(lb=f.bounds[0].cpu().data.numpy(), ub=f.bounds[1].cpu().data.numpy(), dims=f.dim, ninits=10,
                         func=f, args=args,
                         run=args.data_id)
        agent.search()






