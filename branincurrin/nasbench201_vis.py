import json
from utils import latin_hypercube, from_unit_cube, convert_dtype
from botorch.test_functions.multi_objective import VehicleSafety, BraninCurrin, DTLZ2
import torch
tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
# with open('./nasbench201', 'r') as f:
#     nas = json.load(f)
#
#
# n = {}
#
# for key in nas:
#     new_key = []
#     ekey = eval(key)
#     new_key.append(ekey[0])
#     new_key.append(5 * ekey[1] + ekey[2])
#     new_key.append(25 * ekey[3] + 5 * ekey[4] + ekey[5])
#
#     print(new_key)
#     n[str(new_key)] = nas[key]
#
# with open('./nasbench201-3d', 'w') as f:
#     json.dump(n, f)

f = VehicleSafety(negate=True).to(**tkwargs)

init_points = latin_hypercube(10, f.dim)
samples = from_unit_cube(init_points, lb=f.bounds[0].data.numpy(), ub=f.bounds[1].data.numpy())




samples = torch.tensor(samples)
objs = f(samples)

print(samples)
objs = torch.tensor(objs)

print(objs)

with open('./nasbench201-3d', 'r') as f:
    nas = json.load(f)

samples = []
objs = []
for key in nas:
    samples.append(eval(key))
    objs.append(nas[key])


samples = torch.tensor(samples, dtype=torch.float64)
objs = torch.tensor(objs, dtype=torch.float64)
print(samples)
print(objs)


