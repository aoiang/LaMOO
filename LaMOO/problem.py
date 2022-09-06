import json
import numpy as np
import torch

class MyProblem:
    _max_hv = None    ## Type the maximum hypervolume here if it is known
    def __init__(self, dims:int, num_objectives:int, ref_point:torch.tensor, bounds:torch.tensor):
        self.dim = dims                         # problem dimensions, int
        self.num_objectives = num_objectives    # number of objective, int
        self.ref_point = ref_point              # reference point, torch tensor
        self.bounds = bounds                    # bound for dimensions, torch.tensor

        #### Take two dimensions problem for example
        # bounds = [(0.0, 0.99999)] * self.dim
        # self.bounds = torch.tensor(bounds, dtype=torch.float).transpose(-1, -2)

    def __call__(self, x):
        res = MyProblem(x)                      # Here is to customize your function
        return res

