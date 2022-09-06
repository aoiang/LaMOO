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
tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }



f = BraninCurrin(negate=True).to(**tkwargs)
init_cands = latin_hypercube(300000, f.dim)
init_cands = from_unit_cube(init_cands, f.bounds[0].data.numpy(), f.bounds[1].data.numpy())
init_cands = torch.tensor(init_cands)

if not os.path.exists('./state/samples.json'):
    init_points = latin_hypercube(6200, f.dim)
    samples = from_unit_cube(init_points, lb=f.bounds[0].data.numpy(), ub=f.bounds[1].data.numpy())
    sams = {}
    sams['samples'] = samples.tolist()
    with open('./state/samples.json', 'w') as file:
        json.dump(sams, file)

else:
    with open('./state/samples.json', 'r') as file:
        samples = json.load(file)['samples']

samples = torch.tensor(samples)

print(type(samples))
print()

objs = f(samples)
objs = torch.tensor(objs)
pareto_mask = is_non_dominated(objs)
x_pareto = samples[pareto_mask]



# print(x_pareto)
x_pareto = x_pareto.cpu().data.numpy()




new_x = []
for x in x_pareto:
    if x[0] < 0.4:
        new_x.append(x.tolist())

x_pareto = torch.tensor(new_x, dtype=torch.float64)

y_pareto = f(x_pareto)

# print(x_pareto)


def plot1(t=26, spec=None):

    samples = []
    for iter in range(t):
        if spec is not None:
            if iter < spec:
                continue
        with open('./state/selfnodes_' + 'iter_' + str(iter) + '.pkl', 'rb') as f:
            selfnodes = pickle.load(f)

        cur_bag = []
        for node in selfnodes:
            if node.is_leaf():
                cur_bag.extend(node.bag[1].cpu().data.numpy())

        if iter == 0:
            samples = torch.tensor(cur_bag)
        else:
            valid_bag = []
            for s1 in cur_bag:
                valid = True
                for s2 in samples:
                    if torch.equal(torch.tensor(s1), s2):
                        valid = False
                        break
                if valid:
                    # print(s1)
                    valid_bag.append(s1)



            valid_bag = torch.tensor(valid_bag)
            # print(valid_bag)
            samples = torch.cat([samples, valid_bag])

        print('this is iter ', iter)
        print(samples)







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
    color_id = 0
    color_list = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf', 'silver', 'plum', 'sienna', 'chartreuse', 'black', 'ivory']

    batch_number = torch.cat(
        [torch.zeros(10), torch.arange(1, 10 + 1).repeat(2, 1).t().reshape(-1)]
    ).numpy()

    plt.scatter(
        samples[:, 0].cpu().numpy(), samples[:, 1].cpu().numpy(), c=batch_number, alpha=0.8, cmap='RdBu_r',
    )

    ax.set_xlabel("Objective 1")

    # plt.xlim(-260, 5)
    # plt.ylim(-15, 0)
    ax.set_ylabel("Objective 2")
    norm = plt.Normalize(batch_number.min(), batch_number.max())
    # sm = ScalarMappable(norm=norm, cmap=cm)
    sm = ScalarMappable(norm=norm, cmap='RdBu_r')
    # sm.set_array([])
    # fig.subplots_adjust(right=0.9)

    cbar_ax = fig.add_axes([0.935, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.set_title("Iter")



    plt.show()

    # fig.savefig('./figs/' + 'iter_' + str(iter) +'_region_split_.png', bbox_inches='tight')

def plot2(t=26, spec=None):
    for iter in range(t):
        if spec is not None:
            if iter < spec:
                continue
        print('this is iter--------------------------------', iter)
        with open('./state/selfnodes_' + 'iter_' + str(iter) + '.pkl', 'rb') as f:
            selfnodes = pickle.load(f)

        f = BraninCurrin(negate=True).to(**tkwargs)
        s = latin_hypercube(1, f.dim)
        s = from_unit_cube(s, f.bounds[0].data.numpy(), f.bounds[1].data.numpy())
        s = torch.tensor(s)
        for node in selfnodes:
            s = torch.cat([s, node.bag[0]])

        x1o_min, x1o_max = -300.0, 0.0
        x2o_min, x2o_max = -15.0, -0.0

        h1 = 1.5
        h2 = 0.075

        xx_obj, yy_obj = np.meshgrid(np.arange(x1o_min, x1o_max, h1),
                             np.arange(x2o_min, x2o_max, h2))

        x1_min, x1_max = 0.0, 1.0
        x2_min, x2_max = 0.0, 1.0

        h = 0.005
        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                             np.arange(x2_min, x2_max, h))

        mesh_samples = np.c_[xx.ravel(), yy.ravel()]
        mesh_samples_obj = f(torch.tensor(mesh_samples)).cpu().data.numpy()



        node_samples_index = {}
        whole_index = []
        for i in range(len(mesh_samples)):
            whole_index.append(i)
        for node in selfnodes:
            node_samples_index[node.get_name()] = []
        node_samples_index[selfnodes[0].get_name()] = whole_index

        node_mesh_samples = {}
        node_mesh_samples[selfnodes[0].get_name()] = mesh_samples

        nodes = {}
        node_cands = {}
        node_cands[selfnodes[0].get_name()] = s
        # node_cands[selfnodes[0].get_name()] = init_cands

        for node in selfnodes:
            nodes[node.get_name()] = node

        for node in selfnodes:
            path = [node]
            cur_node = node
            while cur_node.parent:
                path.insert(0, nodes[cur_node.get_parent_str()])
                cur_node = nodes[cur_node.get_parent_str()]

            for p in range(len(path)):
                if path[p].get_name() not in node_cands:
                    if path[p].is_good_kid():
                        boundary = path[p - 1].classifier.svm
                        cands = node_cands[path[p - 1].get_name()][
                            boundary.predict(node_cands[path[p - 1].get_name()]) == 0]
                        node_cands[path[p].get_name()] = cands

                    else:
                        boundary = path[p - 1].classifier.svm
                        cands = node_cands[path[p - 1].get_name()][
                            boundary.predict(node_cands[path[p - 1].get_name()]) == 1]
                        node_cands[path[p].get_name()] = cands

        total = 0
        for node in selfnodes:
            if not node.is_leaf():
                index_0 = (node.classifier.svm.predict(node_mesh_samples[node.get_name()]) == 0)
                index_1 = (node.classifier.svm.predict(node_mesh_samples[node.get_name()]) == 1)
                node_mesh_samples[node.kids[0].get_name()] = node_mesh_samples[node.get_name()][index_0]
                node_mesh_samples[node.kids[1].get_name()] = node_mesh_samples[node.get_name()][index_1]

                # print('index_0 is', index_0)
                # print('index_1 is', index_1)

                assert len(index_0) == len(node_samples_index[node.get_name()])
                for i in range(len(index_0)):
                    if index_0[i]:
                        node_samples_index[node.kids[0].get_name()].append(node_samples_index[node.get_name()][i])
                    else:
                        node_samples_index[node.kids[1].get_name()].append(node_samples_index[node.get_name()][i])

                assert len(node_samples_index[node.kids[0].get_name()]) == len(node_mesh_samples[node.kids[0].get_name()])
                assert len(node_samples_index[node.kids[1].get_name()]) == len(node_mesh_samples[node.kids[1].get_name()])

            else:
                total += len(node_samples_index[node.get_name()])






        from matplotlib import rcParams

        rcParams['font.family'] = 'sans-serif'
        # rcParams['image.cmap'] = 'viridis'
        plt.rc('font', size=20)
        plt.rc('axes', labelsize=20)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        plt.rc('legend', fontsize=20)
        fig, ax = plt.subplots(figsize=(7, 5))
        color_id = 0
        color_list = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                      '#17becf', 'silver', 'plum', 'sienna', 'chartreuse', 'black', 'ivory']




        plt.scatter(y_pareto[:, 0], y_pareto[:, 1], color='red', marker='*', label='Pareto frontier')




        for node in selfnodes:
            if node.is_leaf():
                best = True
                cur_node = node
                while cur_node.get_name() != 'node0':
                    if not cur_node.is_good_kid():
                        best = False
                        break
                    else:
                        cur_node = cur_node.parent

                if best:

                    print('best node is', node.get_name())
                    print('cur_color id is', 0)
                    # print('x is', node_cands[node.get_name()][:, 0])
                    # plt.scatter(node_cands[node.get_name()][:, 0], node_cands[node.get_name()][:, 1],
                    #             cmap=plt.cm.coolwarm, label=node.get_name() + '_is_selected_node')

                    # plt.scatter(node_cands[node.get_name()][:, 0], node_cands[node.get_name()][:, 1],
                    #             label=node.get_name() + '(selected)', c='#1f77b4', alpha=0.0)

                    obj_space_samples = f(node_cands[node.get_name()])
                    plt.scatter(obj_space_samples[:, 0], obj_space_samples[:, 1], facecolor='#1f77b4', lw=0, alpha=1)


                    # plt.scatter(node_cands[node.get_name()][:, 0], node_cands[node.get_name()][:, 1],
                    #             label=node.get_name() + '(selected)', c='#1f77b4', alpha=1.0)
                    # plt.scatter(nodes[node.get_name()].bag[0][:, 0], nodes[node.get_name()].bag[0][:, 1], c='#1f77b4')


                else:
                    # print('x is', node_cands[node.get_name()][:, 0])

                    print('cur_node is', node.get_name())
                    print('cur_color id is', color_id+1)
                    obj_space_samples = f(node_cands[node.get_name()]).cpu().data.numpy()




                    plt.scatter(obj_space_samples[:, 0], obj_space_samples[:, 1],
                                c=color_list[color_id], lw=0, alpha=1)



                    # plt.scatter(node_cands[node.get_name()][:, 0], node_cands[node.get_name()][:, 1],
                    #         c=color_list[color_id], label=node.get_name(), alpha=0.5)
                    #     plt.scatter(nodes[node.get_name()].bag[0][:, 0], nodes[node.get_name()].bag[0][:, 1], c=color_list[color_id],
                    #             alpha=1)
                    color_id += 1

        # plt.scatter(self.samples[0][:, 0], self.samples[0][:, 1], c='black', marker='*')
        plt.legend()

        # plt.xlim(0.0, 1.0)
        # plt.ylim(0.0, 1.0)

        # plt.xticks(())
        # plt.yticks(())
        plt.xlabel('objective1')
        plt.ylabel('objective2')

        fig.show()
        # fig.savefig('./figs/' + 'iter_' + str(iter) + 'leaf_objs.png', bbox_inches='tight')

plot1(11)
# plot2(11, spec=10)