# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 23:51:05 2019

@author: Ananthan
"""

import time
import math
import pandas as pd

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def sort_csv(csv_path):
    df=pd.read_csv(csv_path)
    df=df.sort_values("ids")
    df.to_csv("sorted_"+csv_path,index=False)