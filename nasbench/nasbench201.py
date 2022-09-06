import json
import numpy as np
import torch

class Nasbench201:
    _max_hv = 8.06987476348877
    discrete = True
    def __init__(self):
        self.dim = 6
        self.num_objectives = 2
        if torch.cuda.is_available():
            self.ref_point = torch.tensor([-3.0, -6.0], device='cuda')

        else:
            self.ref_point = torch.tensor([-3.0, -6.0])

        bounds = [(0.0, 0.99999)] * self.dim

        self.bounds = torch.tensor(bounds, dtype=torch.float).transpose(-1, -2)

        with open('std_normalized_nasbench201', 'r') as f:
            self.dataset = json.load(f)

    def __call__(self, x):
        x = x.tolist()
        for i in range(len(x)):
            for j in range(len(x[i])):


                if 0 <= x[i][j] < 0.2:
                    x[i][j] = 0.0
                elif 0.2 <= x[i][j] < 0.4:
                    x[i][j] = 0.2
                elif 0.4 <= x[i][j] < 0.6:
                    x[i][j] = 0.4
                elif 0.6 <= x[i][j] < 0.8:
                    x[i][j] = 0.6
                else:
                    x[i][j] = 0.8

        res = []
        for arch in x:
            res.append([(self.dataset[str(arch)][0] - 2), -(self.dataset[str(arch)][1] + 2)])
        if torch.cuda.is_available():
            res = torch.tensor(res, dtype=torch.float, device='cuda')
        else:
            res = torch.tensor(res, dtype=torch.float)
        return res

    def encode_to_nasbench201(self, sample):
        sample = sample.cpu().data.numpy().tolist()
        # print(sample)
        for i in range(len(sample)):
            if 0 <= sample[i] < 0.2:
                sample[i] = 0.0
            elif 0.2 <= sample[i] < 0.4:
                sample[i] = 0.2
            elif 0.4 <= sample[i] < 0.6:
                sample[i] = 0.4
            elif 0.6 <= sample[i] < 0.8:
                sample[i] = 0.6
            else:
                sample[i] = 0.8
        return str(sample)

