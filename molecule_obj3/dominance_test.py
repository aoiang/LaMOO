import matplotlib.pyplot as plt
import numpy as np
from utils import latin_hypercube, from_unit_cube, convert_dtype
# from botorch.test_functions.multi_objective import VehicleSafety, BraninCurrin, DTLZ2
import torch
import botorch
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume

import pygmo as pg
from copy import deepcopy
from moo_molecule_funcs.properties import MOOMoleculeFunction
from moo_molecule_funcs.properties import SUPPORTED_PROPERTIES

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}

# f = BraninCurrin(negate=True).to(**tkwargs)
f = MOOMoleculeFunction(list(SUPPORTED_PROPERTIES.keys()))


botorch_hv = Hypervolume(ref_point=torch.tensor(f.ref_point))
obj = []
for i in range(36):
    init_points = latin_hypercube(100, 32)
    init_points = from_unit_cube(init_points, f.bounds[0].data.numpy(), f.bounds[1].data.numpy())

    samples = torch.tensor(init_points)
    try:
        obj.append(f(samples).cpu())
    except:
        continue

# print('obj is', obj)
obj = torch.cat(obj, 0)

obj_list = deepcopy(obj.numpy())
obj_list *= -1

# obj_list *= -1
# print('obj is', obj_list)

ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=obj_list)
print(dc)

id_dict = {}
for i in range(len(dc)):
    id_dict[i] = dc[i]

sorted_domi = np.array(sorted(id_dict.items(), key=lambda kv:(kv[1])))[:, 0]

good_samples = [False] * len(sorted_domi)

for i in range(len(sorted_domi)):
    if i <= (len(sorted_domi) // 2):
        good_samples[int(sorted_domi[i])] = True

good_samples = torch.tensor(good_samples)

print(good_samples)

pareto_mask = is_non_dominated(obj)
pareto_y = obj[pareto_mask]
hv = botorch_hv.compute(pareto_y)





print('hv is', hv)
fig, ax = plt.subplots(figsize=(7, 5))

# bpareto_x = samples[pareto_mask].data.numpy()
pareto_y = pareto_y.data.numpy()

pareto_obj = obj[pareto_mask].data.numpy()
print(pareto_mask)

good_obj = obj[good_samples].data.numpy()




two_list = deepcopy(good_obj)
two_list *= -1

ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=two_list)
print(dc)

id_dict = {}
for i in range(len(dc)):
    id_dict[i] = dc[i]

sorted_domi = np.array(sorted(id_dict.items(), key=lambda kv:(kv[1])))[:, 0]

good_samples = [False] * len(sorted_domi)

for i in range(len(sorted_domi)):
    if i <= (len(sorted_domi) // 2):
        good_samples[int(sorted_domi[i])] = True

good_samples = torch.tensor(good_samples)
two_obj = good_obj[good_samples]




three_list = deepcopy(two_obj)
three_list *= -1

ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=three_list)
print(dc)

id_dict = {}
for i in range(len(dc)):
    id_dict[i] = dc[i]

sorted_domi = np.array(sorted(id_dict.items(), key=lambda kv:(kv[1])))[:, 0]

good_samples = [False] * len(sorted_domi)

for i in range(len(sorted_domi)):
    if i <= (len(sorted_domi) // 2):
        good_samples[int(sorted_domi[i])] = True

good_samples = torch.tensor(good_samples)
three_obj = two_obj[good_samples]


four_list = deepcopy(three_obj)
four_list *= -1

ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=four_list)
print(dc)

id_dict = {}
for i in range(len(dc)):
    id_dict[i] = dc[i]

sorted_domi = np.array(sorted(id_dict.items(), key=lambda kv:(kv[1])))[:, 0]

good_samples = [False] * len(sorted_domi)

for i in range(len(sorted_domi)):
    if i <= (len(sorted_domi) // 2):
        good_samples[int(sorted_domi[i])] = True

good_samples = torch.tensor(good_samples)
four_obj = three_obj[good_samples]



five_list = deepcopy(four_obj)
five_list *= -1

ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=five_list)
print(dc)

id_dict = {}
for i in range(len(dc)):
    id_dict[i] = dc[i]

sorted_domi = np.array(sorted(id_dict.items(), key=lambda kv:(kv[1])))[:, 0]

good_samples = [False] * len(sorted_domi)

for i in range(len(sorted_domi)):
    if i <= (len(sorted_domi) // 2):
        good_samples[int(sorted_domi[i])] = True

good_samples = torch.tensor(good_samples)
five_obj = four_obj[good_samples]






# plt.scatter(samples[:, 0], samples[:, 1], color='darkred')
#
# plt.xlim(0, 1)
# plt.ylim(0, 1)
#
#
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()

plt.scatter(obj[:, 0], obj[:, 1], label='all')
plt.scatter(good_obj[:, 0], good_obj[:, 1], label='1st')
plt.scatter(two_obj[:, 0], two_obj[:, 1], label='2nd')
plt.scatter(three_obj[:, 0], three_obj[:, 1], label='3rd')
plt.scatter(four_obj[:, 0], four_obj[:, 1], label='4th')
plt.scatter(five_obj[:, 0], five_obj[:, 1], label='5th')
plt.scatter(pareto_obj[:, 0], pareto_obj[:, 1], label='pareto')

plt.legend()
plt.xlabel("F1(x)")
plt.ylabel("F2(x)")
fig.savefig('good_obj.pdf', bbox_inches='tight')
# plt.show()
