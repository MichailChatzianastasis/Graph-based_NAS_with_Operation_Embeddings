import numpy as np
from sklearn.manifold import TSNE
from util import *
import torch
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
import scipy.stats as sps
from scipy.io import loadmat
from scipy.stats.stats import pearsonr,spearmanr
import matplotlib.pyplot as plt
from models import *
from bayesian_optimization.evaluate_BN import Eval_BN
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import seaborn as sns


parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
parser.add_argument('--splitting_point', default='1000', help='graph dataset name')
args = parser.parse_args()


with open("igraphs_train.pickle",'rb') as f:
    G = pickle.load(f)
    igraphs=[]
    y=[]
    for g in G:
        igraphs.append(g[0])
        y.append(g[1])
    igraphs=np.array(igraphs)
    y=np.array(y)
    inds =y.argsort()
    sorted_igraphs = igraphs[inds]
    sorted_y=np.sort(y)

    #check graph properties
    diameters=[]
    densities=[]
    transitivity_undirected=[]
    avg=[]
    motifs=[]
    for g in sorted_igraphs:
        diameters.append(g.diameter())
        densities.append(g.density())
        transitivity_undirected.append(g.transitivity_undirected())
        avg.append(g.average_path_length())
        motifs.append(g.motifs_randesu_no())
    plt.scatter(sorted_y,diameters)
    plt.savefig("diameters.png")
    plt.clf()
    plt.scatter(sorted_y,densities)
    plt.savefig("densities.png")
    plt.clf()
    plt.scatter(sorted_y,transitivity_undirected)
    plt.savefig("transitivity.png")
    plt.clf()
    plt.scatter(sorted_y,avg)
    plt.savefig("avg.png")
    plt.clf()
    plt.scatter(sorted_y,motifs)
    plt.savefig("motifs.png")
    

    low_sum=0
    high_sum=0
    for g in sorted_igraphs[:1000]:
        low_sum+=g.transitivity_undirected()
    for g in sorted_igraphs[-1000:]:
        high_sum+=g.transitivity_undirected()
    low_sum=low_sum/1000
    high_sum=high_sum/1000


    
    pear=pearsonr([g.average_path_length() for g in sorted_igraphs],sorted_y)
    print("pearaverage",pear)
    spear=spearmanr([g.average_path_length() for g in sorted_igraphs],sorted_y)

    print("spearmanave",spear)
    pear2=pearsonr([g.transitivity_undirected() for g in sorted_igraphs],sorted_y)
    print(pear2,"pear2")
    spear2=spearmanr([g.transitivity_undirected() for g in sorted_igraphs],sorted_y)
    print("spear2",spear2)
    exit()
    #plt.plot([100,1000,2000,5000,8000],diff)    
    #plt.xlabel('splitting point')
    #plt.ylabel('difference in av.path.length')
    #plt.savefig('diff_path_length')
    positions=[0.7]
    i=0.01
    violins=[]
    curr_violin=[]
    
    trans_violin=[]
    trans_curr_violin=[]

    assort_violin=[]
    assort_curr_violin=[]
    
    clique_number_violin=[]
    clique_number_curr_violin=[]

    for index,acc in enumerate(sorted_y):
        curr_violin.append(sorted_igraphs[index].average_path_length())
        trans_curr_violin.append(sorted_igraphs[index].transitivity_undirected())
        assort_curr_violin.append(sorted_igraphs[index].assortativity_degree())
        clique_number_curr_violin.append(sorted_igraphs[index].clique_number())
        
        if(acc>(0.70+i)):
            violins.append(curr_violin)
            trans_violin.append(trans_curr_violin)
            assort_violin.append(assort_curr_violin)
            clique_number_violin.append(clique_number_curr_violin)

            curr_violin=[]
            trans_curr_violin=[]
            assort_curr_violin=[]
            clique_number_curr_violin=[]

            positions.append(0.70+i)
            i+=0.01
    violins.append(curr_violin)
    trans_violin.append(trans_curr_violin)
    assort_violin.append(assort_curr_violin)
    clique_number_violin.append(clique_number_curr_violin)

    plt.clf()
    colors_list = ['#78C850', '#F08030',  '#6890F0',  '#A8B820',  '#F8D030', '#E0C068', '#C03028', '#F85888', '#98D8D8']
    ax = sns.violinplot(data=violins,palette=colors_list)
    ax.set_xticklabels(positions)
    ax.set_title("Violin Points")
    ax.set_ylabel("Average Path Length")
    ax.set_xlabel("Accuracy")
    plt.savefig('sorted_2d_violin')
   
    plt.clf()
    ax=sns.violinplot(data=trans_violin,palette=colors_list)
    ax.set_xticklabels(positions)
    ax.set_title("Violin Points")
    ax.set_ylabel("Transitivity")
    plt.savefig('trans_violin')

    plt.clf()
    ax=sns.violinplot(data=assort_violin,palette=colors_list)
    ax.set_xticklabels(positions)
    ax.set_title("Violin Points")
    ax.set_ylabel("Assortativity")
    plt.savefig('assort_violin')

    plt.clf()
    ax=sns.violinplot(data=clique_number_violin,palette=colors_list)
    ax.set_xticklabels(positions)
    ax.set_title("Violin Points")
    ax.set_ylabel("Clique number")
    plt.savefig('clique_number_violin')

