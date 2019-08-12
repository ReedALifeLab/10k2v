# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:25:32 2019

@author: Ananthan
"""

import data_processing as dp
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
from util import timeSince
import numpy as np
import os
import pandas as pd
#import plots
import sims

from constants import DATA_PATH, MODEL_FILE,EMBEDDING_DIM



def t_n_d(file):
    """
    gets title and description from a file
    """
    s = file.read()
    des = s[s.index(','):]
    t = s[:s.index(',')]
    return t,des

def train_vectors(data_path, model_file):
    """
    Trains a doc2vec model from character sequences split into 3-letter
    words, then saves it to a file.
    
    """
    start = time.time()
    print("getting data")
    ids,sentences_ls=dp.get_paragraphs(data_path)
    print("got processed data")
    tagged_data = [TaggedDocument(words=dp.preprocess_str_hp(_d).split(), tags=[str(i)]) for i, _d in sentences_ls]
    print("made ", len(sentences_ls), " sentences")
    embedding=build_model(tagged_data)
    print("made model in ",timeSince(start))
    embedding.save(model_file)
    return ids,embedding

def build_model(tagged_data,dim=EMBEDDING_DIM,epochs=20,lr=0.025,dm_model=1):
    model = Doc2Vec(vector_size=dim,
                alpha=lr, 
                min_alpha=lr,
                min_count=1,
                dm =1,
                epochs=1
                )
    print(model.epochs)
    model.build_vocab(tagged_data)
    for epoch in range(epochs):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data,total_examples=model.corpus_count,epochs=model.epochs)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    return model

def make_vec_df(model,path, prefix):
    data=[]
    firms,sent_ls = dp.get_paragraphs(path)
    i=1
    for title,des in sent_ls:
        # print('.', end='')
        try:
            des_ls = dp.preprocess_str_hp(des).split()
            vec = list(model.infer_vector(des_ls))
            data.append(vec)
        except:
            pass
            # print("missed ", filename)
        i+=1
    data = np.array(data).T.tolist()
    df = pd.DataFrame(data, columns=firms)
    return df

if __name__ == "__main__":
    print("training d2v")
    ids,model=train_vectors(DATA_PATH, MODEL_FILE)
    df=make_vec_df(model,DATA_PATH,"",)
    df.to_csv('SPv7_nouns_D2V_vectors.csv')
    sim_df=sims.run_sims('SPv7_nouns_D2V_vectors.csv', 'SPv7_nouns_doc2vec', "", make_null_mat=False)
    #plots.plot_heat_map(sim_df,'d2v.pdf')