# -*- encoding:utf-8 -*-
from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import Sequential
from keras.models import load_model as K_load_model
from keras.utils import np_utils
from keras.layers import Dense, Input, Flatten, Dropout, LSTM, Merge, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge
from keras.models import Model
from utilities import my_callbacks
import argparse
from data_helper import compute_recall_ks, str2bool
#import cPickle

def main():
    
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument('--emb_dim', type=str, default=300, help='Embeddings dimension')
    parser.add_argument('--hidden_size', type=int, default=300, help='Hidden size')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=50, help='Num epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--n_recurrent_layers', type=int, default=1, help='Num recurrent layers')
    parser.add_argument('--input_dir', type=str, default='./dataset/', help='Input dir')
    parser.add_argument('--save_model', type='bool', default=True, help='Whether to save the model')
    parser.add_argument('--model_fname', type=str, default='model/dual_encoder_lstm_classifier.h5', help='Model filename')
    parser.add_argument('--embedding_file', type=str, default='embeddings/glove.840B.300d.txt', help='Embedding filename')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    args = parser.parse_args()
    print ('Model args: ', args)
    np.random.seed(args.seed)
    
    
    print("No pre-trained model...")
    print("Start building model...")
    
    # first, build index mapping words in the embeddings set
    # to their embedding vector
    
    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(args.embedding_file, 'r')
    for line in f:
        values = line.split()
        word = values[0]
        #coefs = np.asarray(values[1:], dtype='float32')
        
        try:
            coefs = np.asarray(values[1:], dtype='float32')
        except ValueError:
            continue
        embeddings_index[word] = coefs
    f.close()

    print("Now loading UDC data...")
    
    train_c, train_r, train_l = pickle.load(open(args.input_dir + 'train.pkl', 'rb'))
    test_c, test_r, test_l = pickle.load(open(args.input_dir + 'test.pkl', 'rb'))
    dev_c, dev_r, dev_l = pickle.load(open(args.input_dir + 'dev.pkl', 'rb'))
    
    print('Found %s training samples.' % len(train_c))
    print('Found %s dev samples.' % len(dev_c))
    print('Found %s test samples.' % len(test_c))
    
    MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index = pickle.load(open(args.input_dir + 'params.pkl', 'rb'))
    MAX_SEQUENCE_LENGTH = 160
    print("MAX_SEQUENCE_LENGTH: {}".format(MAX_SEQUENCE_LENGTH))
    print("MAX_NB_WORDS: {}".format(MAX_NB_WORDS))
    
    
    
    print("Now loading embedding matrix...")
    num_words = min(MAX_NB_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words , args.emb_dim))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print("Now building dual encoder lstm model...")
    # define lstm for sentence1
    m_model = Sequential()
    m_model.add(Embedding(output_dim=args.emb_dim,
                            input_dim=MAX_NB_WORDS,
                            input_length=MAX_SEQUENCE_LENGTH,
                            weights=[embedding_matrix],
                            mask_zero=True,
                            trainable=True))
    
    m_model.add(LSTM(units=args.hidden_size))
    
    context_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    response_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    # these two models will share eveything from shared_cnn
    context_branch = m_model(context_input)
    response_branch = m_model(response_input)
    
    concatenated = merge([context_branch, response_branch], mode='mul')
    out = Dense((1), activation = "sigmoid") (concatenated)

    model = Model([context_input, response_input], out)
    model.compile(loss='binary_crossentropy',
                    optimizer=args.optimizer)
    
    print(m_model.summary())
    print(model.summary())
    
    print("Now training the model...")
    
    histories = my_callbacks.Histories()
    
    bestAcc = 0.0
    patience = 0 
    
    print("\tbatch_size={}, nb_epoch={}".format(args.batch_size, args.n_epochs))
    
    for ep in range(1, args.n_epochs):
        
        #model.fit([train_c, train_r], train_l,
                #batch_size=args.batch_size, nb_epoch=1, callbacks=[histories],
                #validation_data=([dev_c, dev_r], dev_l), verbose=1)
                
        model.fit([train_c, train_r], train_l,
                batch_size=args.batch_size, epochs=1, callbacks=[histories],
                validation_data=([dev_c, dev_r], dev_l), verbose=1)
        
        #model.fit([train_c[:100], train_r[:100]], train_l[:100],
                #batch_size=args.batch_size, epochs=1, callbacks=[histories],
                #validation_data=([dev_c[:100], dev_r[:100]], dev_l[:100]), verbose=1)

        #model.save(model_name + "_ep." + str(ep) + ".h5")

        curAcc =  histories.accs[0]
        if curAcc >= bestAcc:
            bestAcc = curAcc
            patience = 0
        else:
            patience = patience + 1

        #doing classify the test set
        y_pred = model.predict([test_c, test_r])        
        #y_pred = model.predict([test_c[:100], test_r[:100]])     
        
        print("Perform on test set after Epoch: " + str(ep) + "...!")    
        recall_k = compute_recall_ks(y_pred[:,0])
        
        #stop the model whch patience = 10
        if patience > 10:
            print("Early stopping at epoch: "+ str(ep))
            break
        
    if args.save_model:
        print("Now saving the model... at {}".format(args.model_fname))
        model.save(args.model_fname)

if __name__ == "__main__":
    main()
    