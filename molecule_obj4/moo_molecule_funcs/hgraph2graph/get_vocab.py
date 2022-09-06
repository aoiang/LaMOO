import sys
import argparse 
from hgraph import *
from rdkit import Chem
from multiprocessing import Pool

from props.properties import qed, drd2, penalized_logp
from finetune_generator import Chemprop

def process(data):
    data, prop = data
    if prop is not None:
        values = set()
        if prop == 'qed':
            func = qed
        elif prop == 'drd2':
            func = drd2
        elif prop == 'logp':
            func = penalized_logp
        elif prop == 'SARS':
            evaluator = Chemprop('../SARS-single')
            func = evaluator.predict
        elif prop == 'Antibiotic':
            evaluator = Chemprop('../antibiotics-single')
            func = evaluator.predict
        elif prop == 'bace':
            evaluator = Chemprop('../bace')
            func = evaluator.predict
        elif prop == 'bbbp':
            evaluator = Chemprop('../bbbp')
            func = evaluator.predict
        elif prop == 'hiv':
            evaluator = Chemprop('../hiv')
            func = evaluator.predict

        if prop in ['SARS', 'Antibiotic', 'bace', 'bbbp', 'hiv']:
            smiles = []
            for line in data:
                s = line.strip("\r\n ")
                smiles.append(s)
            preds = func(smiles)
            assert len(preds) == len(smiles)
            for s, v in zip(smiles, preds):
                values.add((s, v))
        else:
            for line in data:
                s = line.strip("\r\n ")
                values.add((s, func(s)))
        return values
    vocab = set()
    for line in data:
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        for node,attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add( attr['label'] )
            for i,s in attr['inter_label']:
                vocab.add( (smiles, s) )
    return vocab

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=1)
    parser.add_argument('--prop', type=str, default=None, choices=['qed', 'drd2', 'SARS', 'Antibiotic', 'logp', 'bace', 'bbbp', 'hiv'])
    args = parser.parse_args()

    data = [mol for line in sys.stdin for mol in line.split()[:2]]
    data = list(set(data))

    batch_size = len(data) // args.ncpu + 1
    batches = [(data[i : i + batch_size], args.prop) for i in range(0, len(data), batch_size)]

    # if args.prop in ['SARS', 'Antibiotic', 'bace', 'bbbp', 'hiv']:
    #     vocab = process((data, args.prop))
    #     vocab = list(set(vocab))
    # else:
    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, batches)
    vocab = [(x,y) for vocab in vocab_list for x,y in vocab]
    vocab = list(set(vocab))

    if args.prop is None:
        for x,y in sorted(vocab):
            print(x, y)
    else:
        for x,y in sorted(vocab, key=lambda v: v[1]):
            print(x, y)
