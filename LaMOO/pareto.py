import matplotlib.patches as mpatches
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils import latin_hypercube, from_unit_cube, convert_dtype
from botorch.test_functions.multi_objective import VehicleSafety, BraninCurrin, DTLZ2
from botorch.utils.multi_objective.pareto import is_non_dominated
import torch
import json
import os
import pygmo as pg
tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }



# f = BraninCurrin(negate=True).to(**tkwargs)
f = VehicleSafety(negate=True).to(**tkwargs)
# init_cands = latin_hypercube(100, f.dim)
# init_cands = from_unit_cube(init_cands, f.bounds[0].data.numpy(), f.bounds[1].data.numpy())

# if not os.path.exists('./state/samples.json'):
init_points = latin_hypercube(1230, f.dim)
samples = from_unit_cube(init_points, lb=f.bounds[0].data.numpy(), ub=f.bounds[1].data.numpy())
sams = {}
sams['samples'] = samples.tolist()
with open('./state/vsamples.json', 'w') as file:
    json.dump(sams, file)

# else:
#     with open('./state/samples.json', 'r') as file:
#         samples = json.load(file)['samples']

samples = torch.tensor(samples)



objs = f(samples)
objs = torch.tensor(objs)

ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(points=objs)


pareto_mask = is_non_dominated(objs)

x_pareto = samples[pareto_mask]
x_samples = torch.tensor(x_pareto)

counter = 0
for i in range(1, len(dc) + 1):
    bag = []
    for j in range(0, len(dc)):
        if dc[j] == i:
            bag.append(samples[j].cpu().numpy())
    if len(bag) > 0:
        counter += 1
        # print(bag)
        bag = torch.tensor(bag)
        x_samples = torch.cat([x_samples, bag])

counter += 1
x_samples = x_samples[:1200]
x_samples = x_samples.cpu().data.numpy()
# x_samples = x_samples[::-1]

print('counter is', counter)
print('length is', len(x_samples))

x_pareto = x_pareto.cpu().data.numpy()
samples = samples.cpu().data.numpy()

new_x = []
for x in x_pareto:
    if x[0] < 0.4:
        new_x.append(x.tolist())

x_pareto = torch.tensor(new_x, dtype=torch.float64)




from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
from matplotlib.cm import ScalarMappable
plt.rc('font', size=20)
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('legend', fontsize=22.5)
cm = plt.cm.get_cmap('viridis')

# plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
fig, ax = plt.subplots(figsize=(7, 5))

batch_number = torch.cat(
    [torch.zeros(40), torch.arange(1, 29 + 1).repeat(len(x_samples) // 30, 1).t().reshape(-1)]
).numpy()

plt.scatter(
    x_samples[:, 0], x_samples[:, 1], c=batch_number, alpha=0.8, cmap='RdBu',
)


norm = plt.Normalize(batch_number.min(), batch_number.max())
# sm = ScalarMappable(norm=norm, cmap=cm)
sm = ScalarMappable(norm=norm, cmap='RdBu')

# sm.set_array([])
# fig.subplots_adjust(right=0.9)

cbar_ax = fig.add_axes([0.935, 0.15, 0.01, 0.7])
cbar = fig.colorbar(sm, cax=cbar_ax)
# cbar.ax.set_title("#Do")
# cbar.ax.set_yticklabels(['low', '', '', '', '', '', 'high'])
cbar.ax.set_yticklabels([])
ax.axis('off')
fig.show()

fig.savefig('pareto_cute.pdf', bbox_inches='tight')







