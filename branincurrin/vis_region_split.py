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
init_cands = latin_hypercube(100000, f.dim)
init_cands = from_unit_cube(init_cands, f.bounds[0].data.numpy(), f.bounds[1].data.numpy())

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
print(x_pareto)

x_pareto = x_pareto.cpu().data.numpy()

new_x = []
for x in x_pareto:
    if x[0] < 0.4:
        new_x.append(x.tolist())

x_pareto = torch.tensor(new_x, dtype=torch.float64)

print(x_pareto)


def plot1(t=26):
    for iter in range(t):
        with open('./state/selfnodes_' + 'iter_' + str(iter) + '.pkl', 'rb') as f:
            selfnodes=pickle.load(f)


        x1_min, x1_max = 0.0, 1.0
        x2_min, x2_max = 0.0, 1.0

        h = 0.001

        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                             np.arange(x2_min, x2_max, h))

        mesh_samples = np.c_[xx.ravel(), yy.ravel()]

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
        node_cands[selfnodes[0].get_name()] = init_cands
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

                assert len(node_samples_index[node.kids[0].get_name()]) == len(
                    node_mesh_samples[node.kids[0].get_name()])
                assert len(node_samples_index[node.kids[1].get_name()]) == len(
                    node_mesh_samples[node.kids[1].get_name()])

            else:
                total += len(node_samples_index[node.get_name()])

        Z = []
        for i in range(len(mesh_samples)):
            Z.append(-1)

        color = 1
        # if iter == 10:
        #     print(node_samples_index['node4'])

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
                # print('length is', len(node_samples_index[node.get_name()]))
                if best:
                #     print('region node name is', node.get_name())
                #     print('color is', color) if not best else print('color is', 0)
                #     for index in node_samples_index[node.get_name()]:
                #         Z[index] = 0
                    cur_node = node
                    while cur_node.get_name() != 'node0':
                        fig, ax = plt.subplots(figsize=(7, 5))
                        Z = []
                        for i in range(len(mesh_samples)):
                            Z.append(-1)

                        color = 1

                        print('region node name is', cur_node.get_name())
                        print('color is', color) if not best else print('color is', 0)
                        for index in node_samples_index[cur_node.get_name()]:
                            Z[index] = 0



                        Z = np.array(Z)
                        Z = Z.reshape(xx.shape)
                        # print(Z)
                        if iter > 1:
                            plt.contourf(xx, yy, Z, color,
                                         colors=['white', '#1f77b4'], alpha=0.5)
                        else:
                            plt.contourf(xx, yy, Z, color,
                                         colors=['#1f77b4', 'white'], alpha=0.5)

                        plt.scatter(x_pareto[:, 0], x_pareto[:, 1], color='red', marker='*',
                                                                label='Pareto frontier')

                        plt.xlim(0.0, 1.0)
                        plt.ylim(0.0, 1.0)

                        if iter == 10 and cur_node.get_name() == 'node1':
                            patch = mpatches.Patch(color='#1f77b4', label='Selected space', alpha=0.5)
                            plt.legend(handles=[patch, a], loc='lower left')


                        fig.savefig('./figs/' + 'iter_' + str(iter) + cur_node.get_name() + '_.png',
                                    bbox_inches='tight')

                        cur_node = cur_node.parent



        from matplotlib import rcParams
        rcParams['font.family'] = 'sans-serif'
        plt.rc('font', size=20)
        plt.rc('axes', labelsize=20)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        plt.rc('legend', fontsize=22.5)

        # plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
        fig, ax = plt.subplots(figsize=(7, 5))
        color_id = 0
        color_list = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                      '#17becf', 'silver', 'plum', 'sienna', 'chartreuse', 'black', 'ivory']

        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        # print(Z)
        if iter > 1:
            plt.contourf(xx, yy, Z, color,
                         colors=['white', '#1f77b4'], alpha=0.5)
        else:
            plt.contourf(xx, yy, Z, color,
                         colors=['#1f77b4', 'white'], alpha=0.5)


        # for node in selfnodes:
        #     if node.is_leaf():
        #         best = True
        #         cur_node = node
        #         while cur_node.get_name() != 'node0':
        #             if not cur_node.is_good_kid():
        #                 best = False
        #                 break
        #             else:
        #                 cur_node = cur_node.parent
        #
        #         if best:
        #
        #             cur_node = node
        #             while cur_node.get_name() != 'node0':
        #
        #                 fig, ax = plt.subplots(figsize=(7, 5))
        #                 ax.set_facecolor('khaki')
        #                 plt.scatter(node_cands[cur_node.get_name()][:, 0], node_cands[cur_node.get_name()][:, 1],
        #                             c='#1f77b4')
        #                 plt.scatter(x_pareto[:, 0], x_pareto[:, 1], color='red', marker='*',
        #                             label='Pareto frontier')
        #                 plt.xlim(0.0, 1.0)
        #                 plt.ylim(0.0, 1.0)
        #
        #                 fig.savefig('./figs/' + 'iter_' + str(iter) + cur_node.get_name() + '_.png',
        #                             bbox_inches='tight')
        #                 cur_node = cur_node.parent
        #
        #
        #             fig, ax = plt.subplots(figsize=(7, 5))
        #
        #             # ax.set_facecolor('khaki')
        #             plt.scatter(node_cands[cur_node.get_name()][:, 0], node_cands[cur_node.get_name()][:, 1],
        #                         c='#1f77b4')
        #             plt.scatter(x_pareto[:, 0], x_pareto[:, 1], color='red', marker='*',
        #                         label='Pareto frontier')
        #             plt.xlim(0.0, 1.0)
        #             plt.ylim(0.0, 1.0)
        #
        #             fig.savefig('./figs/' + 'iter_' + str(iter) + cur_node.get_name() + '_.png',
        #                         bbox_inches='tight')
        #
        #
        #
        #
        #
        #             print('best node is', node.get_name())
        #             print('x is', node_cands[node.get_name()][:, 0])
        #             # plt.scatter(node_cands[node.get_name()][:, 0], node_cands[node.get_name()][:, 1],
        #             #             cmap=plt.cm.coolwarm, label=node.get_name() + '_is_selected_node')
        #             Z = np.array(Z)
        #             fig, ax = plt.subplots(figsize=(7, 5))
        #             Z = Z.reshape(xx.shape)
        #             ax.set_facecolor('khaki')

                    # plt.scatter(node_cands[node.get_name()][:, 0], node_cands[node.get_name()][:, 1], c='#1f77b4')

                # else:
                #     fig2, ax2 = plt.subplots(figsize=(7, 5))
                #     ax2.set_facecolor('khaki')
                #     plt.scatter(node_cands[node.get_name()][:, 0], node_cands[node.get_name()][:, 1], cmap=plt.cm.coolwarm, label=node.get_name())


        # plt.scatter(init_cands[:, 0], init_cands[:, 1], cmap=plt.cm.coolwarm)
        if iter == 0:
            # print("imddfdfd")
            a = plt.scatter(x_pareto[:, 0], x_pareto[:, 1], color='red', marker='*', label='Pareto frontier')
            patch = mpatches.Patch(color='#1f77b4', label='Selected space', alpha=0.5)
            plt.legend(handles=[patch, a], loc='lower left')
        else:
            plt.scatter(x_pareto[:, 0], x_pareto[:, 1], color='red', marker='*')

        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))


        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

        # plt.xticks(())
        # plt.yticks(())

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

        x1_min, x1_max = 0.0, 1.0
        x2_min, x2_max = 0.0, 1.0

        h = 0.0005

        xx, yy = np.meshgrid(np.arange(x1_min, x1_max, h),
                             np.arange(x2_min, x2_max, h))

        mesh_samples = np.c_[xx.ravel(), yy.ravel()]

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

        Z = []
        for i in range(len(mesh_samples)):
            Z.append(-1)

        color = 1
        # if iter == 10:
        #     print(node_samples_index['node4'])

        # for node in selfnodes:
        #     if node.is_leaf():
        #         best = True
        #         cur_node = node
        #         while cur_node.get_name() != 'node0':
        #             if not cur_node.is_good_kid():
        #                 best = False
        #                 break
        #             else:
        #                 cur_node = cur_node.parent
        #         # print('length is', len(node_samples_index[node.get_name()]))
        #         if best:
        #             print('region node name is', node.get_name())
        #             print('color is', color) if not best else print('color is', 0)
        #             for index in node_samples_index[node.get_name()]:
        #                 Z[index] = 0

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
                # print('length is', len(node_samples_index[node.get_name()]))
                if not best:
                    print('region node name is', node.get_name())
                    print('color is', color) if not best else print('color is', 0)
                    for index in node_samples_index[node.get_name()]:
                        Z[index] = color
                    # if not best:
                    color += 1





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
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        # print(Z)
        plt.contourf(xx, yy, Z, color-1, colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                      '#17becf', 'silver', 'plum', 'sienna', 'chartreuse', 'black', 'ivory'], alpha=0.5)
        plt.scatter(x_pareto[:, 0], x_pareto[:, 1], color='black', marker='*', label='Pareto frontier')
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

                    # plt.scatter(node_cands[node.get_name()][:, 0], node_cands[node.get_name()][:, 1],
                    #             label=node.get_name() + '(selected)', c='#1f77b4', alpha=1.0)
                    plt.scatter(node_cands[node.get_name()][:, 0], node_cands[node.get_name()][:, 1],
                                c='#1f77b4', alpha=1.0)
                    # plt.scatter(nodes[node.get_name()].bag[0][:, 0], nodes[node.get_name()].bag[0][:, 1], c='#1f77b4')


                else:
                    # print('x is', node_cands[node.get_name()][:, 0])

                    print('cur_node is', node.get_name())
                    print('cur_color id is', color_id+1)
                    # if len(node_cands[node.get_name()][:, 0]) > 0:
                        # plt.scatter(node_cands[node.get_name()][:, 0], node_cands[node.get_name()][:, 1],
                        #             label=node.get_name(), c=color_list[color_id], alpha=1.0)
                    # plt.scatter(node_cands[node.get_name()][:, 0], node_cands[node.get_name()][:, 1],
                    #         c=color_list[color_id], label=node.get_name(), alpha=1.0)
                    plt.scatter(node_cands[node.get_name()][:, 0], node_cands[node.get_name()][:, 1],
                                c=color_list[color_id], alpha=1.0)
                    #     plt.scatter(nodes[node.get_name()].bag[0][:, 0], nodes[node.get_name()].bag[0][:, 1], c=color_list[color_id],
                    #             alpha=1)
                    color_id += 1

        # plt.scatter(self.samples[0][:, 0], self.samples[0][:, 1], c='black', marker='*')
        # plt.legend(loc='upper left', bbox_to_anchor=(-0.03, 1.396), ncol=2)

        plt.legend()





        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

        # plt.xticks(())
        # plt.yticks(())
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        plt.show()

        # fig.savefig('./figs/' + 'iter_' + str(iter) + 'leaf3.png', bbox_inches='tight')

# plot1(11)
# plot2(11, spec=10)