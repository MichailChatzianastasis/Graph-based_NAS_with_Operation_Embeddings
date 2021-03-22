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
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Train Variational Autoencoders for DAGs')
# general settings

parser.add_argument('--data-name', default='final_structures6', help='graph dataset name')
parser.add_argument('--model', default='DVAE', help='model to use: DVAE, SVAE, \
                    DVAE_fast, DVAE_BN, SVAE_oneshot, DVAE_GCN')

parser.add_argument('--save-appendix', default='',
                    help='what to append to data-name as save-name for results')


parser.add_argument('--epochs', default=300)

args = parser.parse_args()
save_name = args.save_appendix
epochs= args.epochs

latent_space_pkl_path = "./results/"+args.data_name+"_"+save_name+"/" + args.data_name + "_latent_epoch"+str(epochs)+".pkl"
tsne_path_to_save = "./results/"+args.data_name+"_"+save_name+"/" + "tsne"+str(epochs)+".pkl"

X_embedded_train = []
X_embedded_test = []

if(os.path.isfile(tsne_path_to_save)):
    pass #notebook juypyter


else : #run the tsne algorithm and contruct the tsne files
    with open(latent_space_pkl_path, 'rb') as f:
        Z_train, Y_train, Z_test, Y_test = pickle.load(f)
        print(Z_train.shape) # (17118,56)
        print(Z_test.shape) # (1902,56)
        print(Y_train.shape) # (17118,)
        print(Y_test.shape) # (1902,)
        print("after load")
        X_embedded_train = TSNE(n_components=2).fit_transform(Z_train)
        X_embedded_test = TSNE(n_components=2).fit_transform(Z_test)

    with open(tsne_path_to_save, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump((X_embedded_train, Y_train, X_embedded_test, Y_test ), f)