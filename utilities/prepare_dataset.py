"""

here we prepare the dataset in form of .pkl files

"""
# -*- encoding:utf-8 -*-
from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical
import pickle
np.random.seed(1337)
import cPickle

BASE_DIR = ''
GLOVE_DIR = BASE_DIR + '../'
TRAIN_POSITIVE_DATA_DIR = BASE_DIR + '../raw_data_context_response/train/positive/'
TRAIN_NEGATIVE_DATA_DIR = BASE_DIR + '../raw_data_context_response/train/negative/'
DEV_DATA_DIR = BASE_DIR + '../raw_data_context_response/dev/'
TEST_DATA_DIR = BASE_DIR + '../raw_data_context_response/test/'

def load_data():
    #global MAX_SEQUENCE_LENGTH
    #global MAX_NB_WORDS
    #global word_index
    # loading train data
    train_c = []  # list of dev contexts
    train_r = []  # list of dev contexts
    train_l = []  # list of label ids
    
    #for name,k in zip(sorted(os.listdir(TRAIN_POSITIVE_DATA_DIR + "context/")), range(10000)):
    for name in sorted(os.listdir(TRAIN_POSITIVE_DATA_DIR + "context/")):
        path = os.path.join(TRAIN_POSITIVE_DATA_DIR+"context/", name)
        reader = open(path)
        lines = reader.read()
        train_c.append(lines)
        reader.close()
        
        
        path = os.path.join(TRAIN_POSITIVE_DATA_DIR+"response/", name)
        reader = open(path)
        lines = reader.read()
        train_r.append(lines)
        reader.close()
        
        train_l.append(1)
        
    #for name,k in zip(sorted(os.listdir(TRAIN_NEGATIVE_DATA_DIR + "context/")), range(10000)):
    for name in sorted(os.listdir(TRAIN_NEGATIVE_DATA_DIR + "context/")):
        path = os.path.join(TRAIN_NEGATIVE_DATA_DIR+"context/", name)
        reader = open(path)
        lines = reader.read()
        train_c.append(lines)
        reader.close()
        
        
        path = os.path.join(TRAIN_NEGATIVE_DATA_DIR+"response/", name)
        reader = open(path)
        lines = reader.read()
        train_r.append(lines)
        reader.close()
        
        train_l.append(0)
        
    # loading test data
    test_c = []  # list of text contexts
    test_r = []  # list of text contexts
    test_l = []  # list of label ids
    index = 0
    
    responses_list = sorted(os.listdir(TEST_DATA_DIR + "response/"))
    #for name,k in zip(sorted(os.listdir(TEST_DATA_DIR + "context/")), range(500)):
    for name in sorted(os.listdir(TEST_DATA_DIR + "context/")):
        for i in range(10):
            path = os.path.join(TEST_DATA_DIR+"context/", name)
            reader = open(path)
            lines = reader.read()
            test_c.append(lines)
            reader.close()
            
            name_ = responses_list[index] # get the name of the corresponding response
            path = os.path.join(TEST_DATA_DIR+"response/", name_)
            reader = open(path)
            lines = reader.read()
            test_r.append(lines)
            reader.close()
            
            if index % 10 == 0:
                test_l.append(1)
            else:
                test_l.append(0)
            index += 1
    
    
    # loading dev data
    dev_c = []  # list of dev contexts
    dev_r = []  # list of dev contexts
    dev_l = []  # list of label ids
    index = 0
    
    responses_list = sorted(os.listdir(DEV_DATA_DIR + "response/"))
    #for name,k in zip(sorted(os.listdir(DEV_DATA_DIR + "context/")), range(500)):
    for name in sorted(os.listdir(DEV_DATA_DIR + "context/")):
        for i in range(10):
            path = os.path.join(DEV_DATA_DIR+"context/", name)
            reader = open(path)
            lines = reader.read()
            dev_c.append(lines)
            reader.close()
            
            name_ = responses_list[index] # get the name of the corresponding response
            path = os.path.join(DEV_DATA_DIR+"response/", name_)
            reader = open(path)
            lines = reader.read()
            dev_r.append(lines)
            reader.close()
            
            if index % 10 == 0:
                dev_l.append(1)
            else:
                dev_l.append(0)
            index += 1
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_c)
    tokenizer.fit_on_texts(train_r)
    tokenizer.fit_on_texts(test_c)
    tokenizer.fit_on_texts(test_r)
    tokenizer.fit_on_texts(dev_c)
    tokenizer.fit_on_texts(dev_r)

    train_c = tokenizer.texts_to_sequences(train_c)
    train_r = tokenizer.texts_to_sequences(train_r)
    test_c = tokenizer.texts_to_sequences(test_c)
    test_r = tokenizer.texts_to_sequences(test_r)
    dev_c = tokenizer.texts_to_sequences(dev_c)
    dev_r = tokenizer.texts_to_sequences(dev_r)

    MAX_SEQUENCE_LENGTH = max([len(seq) for seq in train_c + train_r
                                                    + test_c + test_r
                                                    + dev_c + dev_r])
    
    
    MAX_NB_WORDS = len(tokenizer.word_index) + 1
    word_index = tokenizer.word_index

    print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))
    print("MAX_NB_WORDS: {}".format(MAX_NB_WORDS))

    train_c = pad_sequences(train_c, maxlen=MAX_SEQUENCE_LENGTH)
    train_r = pad_sequences(train_r, maxlen=MAX_SEQUENCE_LENGTH)
    test_c = pad_sequences(test_c, maxlen=MAX_SEQUENCE_LENGTH)
    test_r = pad_sequences(test_r, maxlen=MAX_SEQUENCE_LENGTH)
    dev_c = pad_sequences(dev_c, maxlen=MAX_SEQUENCE_LENGTH)
    dev_r = pad_sequences(dev_r, maxlen=MAX_SEQUENCE_LENGTH)

    # shuffle training set
    indices = np.arange(train_c.shape[0])
    
    np.random.shuffle(indices)
    
    train_c = np.asarray(train_c)
    train_r = np.asarray(train_r)
    train_l = np.asarray(train_l)
    
    train_c = train_c[indices]
    train_r = train_r[indices]
    train_l = train_l[indices]
    
    cPickle.dump([train_c, train_r, train_l], open("dataset/train.pkl", "wb"), protocol=-1)
    cPickle.dump([test_c, test_r, test_l], open("dataset/test.pkl", "wb"), protocol=-1)
    cPickle.dump([dev_c, dev_r, dev_l], open("dataset/dev.pkl", "wb"), protocol=-1)
    
    cPickle.dump([MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index], open("dataset/params.pkl", "wb"), protocol=-1)
    
    #return train_c, train_r, train_l, test_c, test_r, test_l, dev_c, dev_r, dev_l