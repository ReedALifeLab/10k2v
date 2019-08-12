# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:00:48 2019

@author: Ananthan
"""

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import pandas

def plot_heat_map(matrix,path):
    # height = plt.rcParams['font.size']  * matrix.shape[0] / 10
    # width = plt.rcParams['font.size'] * matrix.shape[1] / 10
    sns.set(font_scale=0.002)
    fig, ax = plt.subplots(figsize=(2^15, 2^15))
    p=sns.heatmap(matrix,vmin= 0.0, vmax = 1.0, linewidths=0.0, square=True, xticklabels=True, yticklabels=True)
    p.figure.savefig(path)
    plt.clf()

def run_plots(PLOTPATHS):
    for matname in PLOTPATHS:
        m = pandas.read_csv(matname, index_col=0)
        plot_heat_map(m, matname[:-4] + "_heat.pdf")