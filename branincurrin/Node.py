from Classifier import Classifier
import json
import numpy as np
import math
import operator
from utils import convert_dtype
import torch

class Node:
    OBJ_COUNTER   = 0
    # If a leave holds >= SPLIT_THRESH and is splittable = True, 
    # we split into two new nodes.
    
    def __init__(self, args, parent=None, dims=0, reset_id=False, cp=10):
        # Note: every node is initialized as a leaf,
        # only internal nodes equip with classifiers to make decisions
        self.dims = dims
        self.x_bar = float('inf')
        self.n = 0
        self.uct = 0
        self.classifier = Classifier(args, [], self.dims)
        self.bag = []
        self.is_svm_splittable = False
        self.cp = cp

        #insert curt into the kids of parent
        self.parent = parent
        self.kids = [] # 0:good, 1:bad
        
        #assign id to each nodes
        if reset_id:
            Node.OBJ_COUNTER = 0
        self.id = Node.OBJ_COUNTER        
        Node.OBJ_COUNTER += 1
    
    def update_kids(self, good_kid, bad_kid):
        assert len(self.kids) == 0
        self.kids.append(good_kid)
        self.kids.append(bad_kid)
        # print('==============x_bar===============')
        # print(self.kids[0].x_bar, self.kids[1].x_bar)
        # assert self.kids[0].x_bar > self.kids[1].x_bar
        
    def is_good_kid(self):
        if self.parent is not None:
            if self.parent.kids[0] == self:
                return True
            else:
                return False
        else:
            return False
    
    def is_leaf(self):
        if len(self.kids) == 0:
            return True
        else:
            return False 
            
    def visit(self):
        self.n += 1
        
    def print_bag(self):
        sorted_bag = sorted(self.bag.items(), key=operator.itemgetter(1))
        print("BAG"+"#"*10)
        for item in sorted_bag:
            print(item[0],"==>", item[1])            
        print("BAG"+"#"*10)
        print('\n')
        
    def update_bag(self, samples, ref_point=None):
        assert len(samples) > 0
        assert ref_point is not None
        
        self.bag.clear()
        self.bag.extend(samples)
        # print('self.bag is!!', self.bag, len(self.bag[0]))
        self.classifier.update_samples(self.bag)
        
        if len(self.bag[0]) <= 15:
            self.is_svm_splittable = False
        else:
            self.is_svm_splittable = self.classifier.is_splittable_svm()

        if not self.is_svm_splittable:
            self.x_bar = self.classifier.get_hypervolume(ref_point)
        
        self.n = len(self.bag[0])
        
    def visualize_node(self):
        self.classifier.viz_learned_boundary(self.get_name())
        self.classifier.viz_sample_clusters(self.get_name())
        
    def clear_data(self):
        self.bag.clear()
    
    def get_name(self):
        # state is a list of jsons
        return "node" + str(self.id)
    
    def pad_str_to_8chars(self, ins, total):
        if len(ins) <= total:
            ins += ' '*(total - len(ins) )
            return ins
        else:
            return ins[0:total]
            
    def get_rand_sample_from_bag(self):
        if len( self.bag ) > 0:
            upeer_boundary = len(list(self.bag))
            rand_idx = np.random.randint(0, upeer_boundary)
            return self.bag[rand_idx][0]
        else:
            return None
            
    def get_parent_str(self):
        return self.parent.get_name()
        
    def propose_samples_sobol(self, num_samples, path, lb, ub, problem):
        proposed_X, proposed_obj = self.classifier.propose_rand_samples_sobol(num_samples, path, lb, ub, problem)
        return proposed_X, proposed_obj

    # def propose_samples_cmaes(self, num_samples, path, lb, ub, func, vanilla, samples=None):
    #     proposed_X, proposed_obj = self.classifier.propose_rand_samples_cmaes(num_samples, path, lb, ub,
    #                                                                           init_within_leaf='random', func=func,
    #                                                                           vanilla=vanilla, samples=samples)
    #     return proposed_X, proposed_obj

    def propose_samples_cmaes(self, num_samples, path, lb, ub, func, vanilla, samples=None):
        proposed_X, proposed_obj = self.classifier.propose_rand_samples_cmaes(num_samples, path, lb, ub,
                                                                              init_within_leaf='random', func=func,
                                                                              vanilla=vanilla, samples=samples)

        return proposed_X, proposed_obj

    def propose_samples_Bayesian_Optimization(self, num_samples, path, lb, ub, model, mll, train_obj, func, acq_func='qehvi', cur_iter=0):
        proposed_X, proposed_obj = self.classifier.propose_rand_samples_bayesian(num_samples, path, lb, ub, model, mll, train_obj, func, acq_func=acq_func, cur_iter=cur_iter)
        return proposed_X, proposed_obj

    def viz_node_samples(self, num_samples, path, lb, ub, iter):
        self.classifier.viz_node_sample_region(num_samples, path, lb, ub, iter)

            
    def propose_samples_bo(self, num_samples, path, lb, ub, samples):
        proposed_X = self.classifier.propose_samples_bo(num_samples, path, lb, ub, samples)
        return proposed_X
        
    def propose_samples_turbo(self, num_samples, path, func):
        proposed_X, fX = self.classifier.propose_samples_turbo(num_samples, path, func)
        return proposed_X, fX

    def propose_samples_rand(self, num_samples):
        assert num_samples > 0
        samples = self.classifier.propose_samples_rand(num_samples)
        return samples
    
    def __str__(self):
        name   = self.get_name()
        name   = self.pad_str_to_8chars(name, 7)
        name  += ( self.pad_str_to_8chars( 'is good:' + str(self.is_good_kid() ), 15 ) )
        name  += ( self.pad_str_to_8chars( 'is leaf:' + str(self.is_leaf() ), 15 ) )
        
        val    = 0
        name  += ( self.pad_str_to_8chars( ' val:{0:.4f}   '.format(round(self.get_xbar(), 3) ), 20 ) )
        name  += ( self.pad_str_to_8chars( ' uct:{0:.4f}   '.format(round(self.get_uct(), 3) ), 20 ) )

        name  += self.pad_str_to_8chars( 'sp/n:'+ str(len(self.bag[0]))+"/"+str(self.n), 15 )

        if torch.cuda.is_available():
            upper_bound = np.around(np.max(self.classifier.samples[0].cpu().data.numpy(), axis=0), decimals=2)
            lower_bound = np.around(np.min(self.classifier.samples[0].cpu().data.numpy(), axis=0), decimals=2)
        else:
            upper_bound = np.around( np.max(self.classifier.samples[0].data.numpy(), axis = 0), decimals=2 )
            lower_bound = np.around( np.min(self.classifier.samples[0].data.numpy(), axis = 0), decimals=2 )
        boundary    = ''
        for idx in range(0, self.dims):
            boundary += str(lower_bound[idx])+'>'+str(upper_bound[idx])+' '

        name  += ( self.pad_str_to_8chars( 'bound:' + boundary, 60 ) )

        parent = '----'
        if self.parent is not None:
            parent = self.parent.get_name()
        parent = self.pad_str_to_8chars(parent, 10)
        
        name += (' parent:' + parent)
        
        kids = ''
        kid  = ''
        for k in self.kids:
            kid   = self.pad_str_to_8chars( k.get_name(), 10 )
            kids += kid
        name += (' kids:' + kids)
        
        return name
    

    def get_uct(self):
        Cp = self.cp
        if self.parent == None:
            return float('inf')
        if self.n == 0:
            return float('inf')
        # print('cp part value is', 2 * Cp * math.sqrt(2 * math.log(self.parent.n) / self.n))
        # print('self.n is', self.n)
        # print('self.parent.n is', self.parent.n)
        # print('math.log self.parent.n is', math.log(self.parent.n))
        # print('CP is', Cp)
        return self.x_bar + 2 * Cp * math.sqrt(2 * math.log(self.parent.n) / self.n)
    
    def get_xbar(self):
        return self.x_bar

    def get_n(self):
        return self.n
        
    def train_and_split(self):
        assert len(self.bag) >= 2
        self.classifier.update_samples(self.bag)
        good_kid_data, bad_kid_data = self.classifier.split_data()
        assert len(good_kid_data[0]) + len(bad_kid_data[0]) == len(self.bag[0])
        return good_kid_data, bad_kid_data

    def plot_samples_and_boundary(self, func):
        name = self.get_name() + ".pdf"
        self.classifier.plot_samples_and_boundary(func, name)

    def sample_arch(self):
        if len(self.bag) == 0:
            return None
        net_str = np.random.choice(list(self.bag.keys()))
        del self.bag[net_str]
        return json.loads(net_str )
