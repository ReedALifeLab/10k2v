# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:55:47 2019

@author: Ananthan
"""
#edited to use freq_vec instead of tfdif
#also produces a shuffled version of the similarity matrix for comparision
import pandas as pd
from scipy import spatial as spat
import numpy as np
#import plots as pl

#new imports for shuffled (null hypothesis) matrix:
import random #random module produces pseudorandom numbers, so possibly not good enough for intensive analysis.
#import secrets
def run_sims(vec_path, vec_name, prefix, make_null_mat=False):
    veccsv=pd.read_csv(vec_path, index_col = 0)
    vec_raw = veccsv.values
    vec_raw_t = np.transpose(vec_raw)
    dots = np.dot(vec_raw_t, vec_raw)
    # print(dots)
    vec_norm = np.sqrt(dots.diagonal())[np.newaxis]
    # print(np.dot(np.transpose(vec_norm), vec_norm))
    sim_mat = ((dots) / np.dot(np.transpose(vec_norm), vec_norm))
    out=pd.DataFrame(sim_mat, columns = veccsv.columns, index = veccsv.columns)
    out.to_csv(vec_name + "_similarities.csv")
    print('made ' + vec_name + ' similarites')

    #here's where we do the shuffling:
    if make_null_mat:
        ind = list(out.index)
        random.shuffle(ind)
        out2 = out.reindex(ind, axis='index')
        out2 = out2.reindex(ind, axis='columns')
        out2.to_csv(prefix + "/" + vec_name + "_similarities_null.csv")
        
    return out

def select_sims(sims_path, select_vec_path, vec_name):
    all_sims=pd.read_csv(sims_path, index_col = 0)
    select_vecs=pd.read_csv(select_vec_path, index_col=0)
    select_firms=list(select_vecs.columns)
    sim_mat = all_sims.loc[select_firms][select_firms]
    out=pd.DataFrame(sim_mat, columns = select_firms, index = select_firms)
    out.to_csv(vec_name + "_similarities.csv")
    print('made ' + vec_name + ' similarites')
    return out


        
