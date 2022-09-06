
import pathlib
import os

import numpy as np
import torch

from hgraph import *

from .multiobj_rationale.properties import *
from .drd2_scorer import get_score as drd2_score
# from multiobj_rationale.properties import *
# from drd2_scorer import get_score as drd2_score
from finetune_generator import Chemprop


class GSK3Prop:
    def __init__(self):
        self.func = gsk3_model()
    def __call__(self, smiles):
        return self.func(smiles)

class JNK3Prop:
    def __init__(self):
        self.func = jnk3_model()
    def __call__(self, smiles):
        return self.func(smiles)

class QEDProp:
    def __init__(self):
        self.func = qed_func()
    def __call__(self, smiles):
        return self.func(smiles)

class SAProp:
    def __init__(self):
        self.func = sa_func()
    def __call__(self, smiles):
        return self.func(smiles)

class DRD2Prop:
    def __init__(self):
        self.func = drd2_score
    def __call__(self, smiles):
        return self.func(smiles)

class HIVProp:
    def __init__(self):
        base_path = pathlib.Path(__file__).parent.resolve()
        evaluator = Chemprop(os.path.join(base_path, 'hiv'))
        self.func = evaluator.predict_single
    def __call__(self, smiles):
        return self.func(smiles)

class SARSProp:
    def __init__(self):
        base_path = pathlib.Path(__file__).parent.resolve()
        evaluator = Chemprop(os.path.join(base_path, 'SARS-single'))
        self.func = evaluator.predict_single
    def __call__(self, smiles):
        return self.func(smiles)


SUPPORTED_PROPERTIES = {
    'gsk3': GSK3Prop, 
    'jnk3': JNK3Prop, 
    # 'qed': QEDProp,
    # 'sa': SAProp,
    # 'drd2': DRD2Prop,
    # 'hiv': HIVProp,
    # 'sars': SARSProp,
}


class MOOMoleculeFunction:
    """
    Give it a list of properties from SUPPORTED_PROPERTIES to initialize a function going from a 32-dim pretrained latent space to the desired properties. 
    """
    _max_hv = 0.38
    def __init__(self, props, device='cuda' if torch.cuda.is_available() else 'cpu'):
        for prop in props:
            assert prop in SUPPORTED_PROPERTIES.keys()
        self.prop_funcs = [SUPPORTED_PROPERTIES[prop]() for prop in props]
        self.device = device
        self.dim = 32
        self.num_objectives = 2
        bounds = [(0.0, 1.0)] * self.dim

        self.bounds = torch.tensor(bounds, dtype=torch.float).transpose(-1, -2)
        if torch.cuda.is_available():
            self.ref_point = torch.tensor([0.0, 0.0], device='cuda')
        else:
            self.ref_point = torch.tensor([0.0, 0.0])

        
        class FakeArgs:
            def __init__(self):
                base_path = pathlib.Path(__file__).parent.resolve()
                self.vocab = os.path.join(base_path, 'hgraph2graph/data/chembl/vocab.txt')
                self.model = os.path.join(base_path, 'hgraph2graph/ckpt/chembl-pretrained/model.ckpt')

                self.atom_vocab = common_atom_vocab
                self.rnn_type = 'LSTM'
                self.hidden_size = 250
                self.embed_size = 250
                self.batch_size = 50
                self.latent_size = 32
                self.depthT = 15
                self.depthG = 15
                self.diterT = 1
                self.diterG = 3
                self.dropout = 0.0
        args = FakeArgs()
        vocab = [x.strip("\r\n ").split() for x in open(args.vocab)] 
        args.vocab = PairVocab(vocab, cuda=(self.device=='cuda'))

        model = HierVAE(args).to(self.device) # latent model

        model.load_state_dict(torch.load(args.model, map_location='cpu')[0])
        # model.load_state_dict(torch.load(args.model)[0])
        model.eval()
        self.model = model


    def __call__(self, sample): 
        # NOTE: it's possible for the decoding to fail sometimes, and i'm not sure why. 
        # you can probably wrap it in a try/except and just treat the value as minimum (which is what we did), or however you want to handle it
        try:
            sample = sample.cpu().numpy()
            root_vecs = torch.from_numpy(sample).to(self.device).view(len(sample), -1).float()
        except:
            root_vecs = torch.from_numpy(sample).to(self.device).view(1, -1).float()

        # print(sample)
        # root_vecs = torch.from_numpy(sample).to(self.device).view(len(sample), -1).float()
        # root_vecs = torch.from_numpy(sample).to(self.device).view(1, -1).float()
        # print(root_vecs)
        smiles = self.model.decoder.decode((root_vecs, root_vecs, root_vecs), greedy=True, max_decode_step=150)
        # print(smiles)
        res = []
        for prop_func in self.prop_funcs:
            res.append(prop_func(smiles))
            # print(prop_func(smiles))
        # print(res)
        res = np.array(res)
        res = res.transpose()
        if torch.cuda.is_available():
            res = torch.tensor(res, dtype=torch.float, device='cuda')
        else:
            res = torch.tensor(res, dtype=torch.float)

        # print(res)
        return res

        # print(res)



        # return [prop_func(smiles) for prop_func in self.prop_funcs]

import torch
# import numpy as np
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}

def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx


def latin_hypercube(n_pts, dim):
    """Basic Latin hypercube implementation with center perturbation."""
    """this assumes x in [0, 1]"""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):  # Shuffle the center locataions for each dimension.
        X[:, i] = centers[np.random.permutation(n_pts)]

    # Add some perturbations within each box
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X



# import matplotlib.pyplot as plt
# import pandas as pd
#
#
# from matplotlib import rcParams
# from botorch.utils.multi_objective.pareto import is_non_dominated
# from botorch.utils.multi_objective.hypervolume import Hypervolume

#
# if __name__=='__main__':
#     # func = MOOMoleculeFunction(list(SUPPORTED_PROPERTIES.keys()))
#
#     ### Molecule
#     n = 150
#     problem = MOOMoleculeFunction(list(SUPPORTED_PROPERTIES.keys()))
#     data = []
#     for i in range(50):
#         seed = np.random.randint(int(1e5))
#
#         train_x = latin_hypercube(n, problem.dim)
#
#         if not torch.cuda.is_available():
#             train_x = from_unit_cube(train_x, problem.bounds[0].data.numpy(), problem.bounds[1].data.numpy())
#             train_x = torch.tensor(train_x)
#         else:
#             train_x = from_unit_cube(train_x, problem.bounds[0].cpu().data.numpy(),
#                                      problem.bounds[1].cpu().data.numpy())
#             train_x = torch.tensor(train_x, device='cuda')
#
#         try:
#             train_obj = problem(train_x)
#         except:
#             continue
#         # print(train_x)
#         # print(train_obj)
#         data.append([train_x, train_obj])
#
#     torch.save(data, 'molecule_data.pt')


    # n = 50
    # problem = MOOMoleculeFunction(list(SUPPORTED_PROPERTIES.keys()))
    # data = []
    # obj = []
    # for i in range(600):
    #     seed = np.random.randint(int(1e5))
    #     print('round', i, 'is done')
    #
    #     train_x = latin_hypercube(n, problem.dim)
    #
    #     if not torch.cuda.is_available():
    #         train_x = from_unit_cube(train_x, problem.bounds[0].data.numpy(), problem.bounds[1].data.numpy())
    #         train_x = torch.tensor(train_x)
    #     else:
    #         train_x = from_unit_cube(train_x, problem.bounds[0].cpu().data.numpy(),
    #                                  problem.bounds[1].cpu().data.numpy())
    #         train_x = torch.tensor(train_x, device='cuda')
    #     try:
    #         train_obj = problem(train_x)
    #         obj.append(train_obj)
    #     except:
    #         continue
    # obj = torch.cat(obj, 0)
    # # print(outs1)
    #
    # pareto_mask = is_non_dominated(obj)
    # pareto_y = obj[pareto_mask]
    # print('===================pareto_y is===========================')
    # print(pareto_y)
    # botorch_hv = Hypervolume(ref_point=torch.tensor([0.0, 0.0]))
    # hv = botorch_hv.compute(pareto_y)
    # print('===================HV is===========================')
    # print(hv)
    #
    # obj = obj.numpy()
    #
    #
    # fig, ax = plt.subplots(figsize=(7, 5))
    # plt.scatter(obj[:, 0], obj[:, 1])
    # plt.scatter(pareto_y[:, 0], pareto_y[:, 1])
    # fig.savefig('molecule.png', bbox_inches='tight')











    # print(problem(np.random.rand(32)))