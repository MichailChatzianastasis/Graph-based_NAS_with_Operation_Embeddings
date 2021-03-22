import pdb
import pickle
import gzip
import argparse
import sys
import os
import os.path
import collections
from shutil import copy
import torch
from tqdm import tqdm
import itertools
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sparse_gp import SparseGP
import scipy.stats as sps
import numpy as np
from scipy.io import loadmat
from scipy.stats.stats import pearsonr
sys.path.append('%s/..' % os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, '../')
from util import *


parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings

parser.add_argument('--data-name', default='final_structures6', help='graph dataset name')
parser.add_argument('--model', default='DVAE', help='model to use: DVAE, SVAE, \
                    DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN')

parser.add_argument('--save-appendix', default='',
                    help='what to append to data-name as save-name for results')


args = parser.parse_args()
save_name = args.save_appendix

scores_arch={}
for file_dir in os.listdir("./ENAS_results"):
    start=file_dir.find('_')
    end=file_dir.rfind('_')
    file_name=file_dir[start+1:end]
    if file_name == save_name:
        print(file_dir)
        f=open("./ENAS_results/"+file_dir+"/best_arc_scores.txt")
        for line in f:
            start=line.find(",")
            score=line[start+1:-1]
            arch=line[0:start]
            scores_arch[arch]=score
#sorting
scores_arch= {k: v for k, v in sorted(scores_arch.items(), key=lambda item: item[1] , reverse=True)}
print(scores_arch)
print(len(scores_arch))
#write to file
f = open("./ENAS_results/"+save_name+"_aggregate.txt","w+")
print(scores_arch, file=f)
f.close()
