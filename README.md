# MULTI-OBJECTIVE OPTIMIZATION BY LEARNING SPACE PARTITIONS (LaMOO)

LaMOO is a a novel multi-objective optimizer that learns a model from observed samples to partition the search space and then focus on promising regions that
are likely to contain a subset of the Pareto frontier. So that existing solvers like Bayesian Optimizations (BO) can focus on promising subregions to mitigate the over-exploring issue.

<p align="center">
<img src='./LaMOO_workflow.png' width="800">
</p>

Please reference the following publication when using this package. OpenReview <a href="https://openreview.net/pdf?id=FlwzVjfMryn">link</a>.


```
@inproceedings{
zhao2022multiobjective,
title={Multi-objective Optimization by Learning Space Partition},
author={Yiyang Zhao and Linnan Wang and Kevin Yang and Tianjun Zhang and Tian Guo and Yuandong Tian},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=FlwzVjfMryn}
}
```
## Environment Requirements
```
python >= 3.8, PyTorch >= 1.8.1, gpytorch >= 1.5.1
scipy, Botorch, Numpy, cma
```
### Extra Requirements for [Molecule Discovery](./molecule/)
```
networkx, RDKit >= 2019.03, Chemprop >= 1.2.0, scikit-learn==0.21.3
```

## Run LaMOO in different test functions (1 minute tutorial)

Open the folder corresponding to the problem's name (including [branincurrin](./branincurrin/), [vehiclesafety](./vehiclesafety/), [Nasbench201](./nasbench/), [molecule](./molecule/), [molecule_obj3](./molecule_obj3/), and [molecule_obj4](./molecule_obj4/)). 

### Evaluate Bayesian Optimization with LaMOO as meta-booster. 

```
python MCTS.py --data_id 0
```

You can change the data_id to have different runs. 
