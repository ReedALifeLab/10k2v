# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:29:39 2019

@author: Ananthan
"""
import pandas as pd
import string
from gensim.parsing import (strip_multiple_whitespaces, strip_tags, strip_punctuation,
                            strip_numeric, remove_stopwords, strip_short, stem_text)
import nltk

def get_paragraphs(data_path):
    df = pd.read_csv(data_path)
    ids = [x for x in df['ids'].values]
    id_and_paragraphs = df.values
    return ids, id_and_paragraphs

def to_lowercase(s):
    return s.lower()


def preserve_special_punct(s, spec_punct=['.', '?', '!', ':', ';']):
    punct = string.punctuation
    for c in spec_punct:
        s = s.replace(c, ' ' + c + ' ')
        punct = punct.replace(c, '')
    for c in punct:
        s = s.replace(c, ' ')
    s = strip_multiple_whitespaces(s)
    s = s.replace(' s ', 's ')
    return s

def preprocess(s):
    for filt in FILTERS:
        s = filt(s)
    return s

def preprocess_str_hp(s):
    s_tokens = nltk.word_tokenize(s)
    tags = nltk.pos_tag(s_tokens)
    nouns = [word for word,pos in tags 
             if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]
    s = ' '.join(nouns).lower()
    s = preprocess(s)
    return s

FILTERS = [to_lowercase, strip_multiple_whitespaces, strip_tags, strip_punctuation,
           preserve_special_punct, strip_numeric, remove_stopwords, strip_short, stem_text]
FILTER_KWS = [f.__name__ for f in FILTERS]
FILTER_LOOKUP = dict(zip(FILTER_KWS, FILTERS))