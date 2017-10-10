# -*- encoding:utf-8 -*-
from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical
import pickle
from keras.models import Sequential
from keras.models import load_model as K_load_model
from keras.utils import np_utils
from keras.layers import Dense, Input, Flatten, Dropout, LSTM, Merge, Activation
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from utilities import cnn_callbacks
import cpickle


#BASE_DIR = ''
#GLOVE_DIR = BASE_DIR + '../'
#TRAIN_POSITIVE_DATA_DIR = BASE_DIR + '../raw_data_context_response/train/positive/'
#TRAIN_NEGATIVE_DATA_DIR = BASE_DIR + '../raw_data_context_response/train/negative/'
#DEV_DATA_DIR = BASE_DIR + '../raw_data_context_response/dev/'
#TEST_DATA_DIR = BASE_DIR + '../raw_data_context_response/test/'
#MAX_SEQUENCE_LENGTH = 1000
#MAX_NB_WORDS = 20000
#EMBEDDING_DIM = 300
#word_index = 0

def compute_recall_ks(probas):
    recall_k = {}
    for group_size in [2, 5, 10]:
        recall_k[group_size] = {}
        print ('group_size: %d' % group_size)
        for k in [1, 2, 5]:
            if k < group_size:
                recall_k[group_size][k] = recall(probas, k, group_size)
                print ('recall@%d' % k, recall_k[group_size][k])
    return recall_k

def recall(probas, k, group_size):
    test_size = 10
    n_batches = len(probas) // test_size
    n_correct = 0
    for i in xrange(n_batches):
        batch = np.array(probas[i*test_size:(i+1)*test_size])[:group_size]
        indices = np.argpartition(batch, -k)[-k:]
        if 0 in indices:
            n_correct += 1
    return float(n_correct) / (len(probas) / test_size)


#EMBEDDING_DIM = 300
#LSTM_DIM = 128

#OPTIMIZER = 'adam'
#BATCH_SIZE = 128
#NB_EPOCH = 50

#TRAINED_CLASSIFIER_PATH = "dual_encoder_lstm_classifier.h5"

def main():
    
    parser = argparse.ArgumentParser()
    parser.register('type','bool',str2bool)
    parser.add_argument('--emb_dim', type=str, default=300, help='Embeddings dimension')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--n_epochs', type=int, default=50, help='Num epochs')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    parser.add_argument('--n_recurrent_layers', type=int, default=1, help='Num recurrent layers')
    parser.add_argument('--input_dir', type=str, default='./dataset/', help='Input dir')
    parser.add_argument('--save_model', type='bool', default=True, help='Whether to save the model')
    parser.add_argument('--model_fname', type=str, default='model/dual_encoder_lstm_classifier.h5', help='Model filename')
    parser.add_argument('--embedding_file', type=str, default='embeddings/glove.840B.300d.txt', help='Embedding filename')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    args = parser.parse_args()
    print 'args:', args
    np.random.seed(args.seed)
    
    #EMBEDDING_DIM = args.emb_dim
    #LSTM_DIM = args.hidden_size

    #OPTIMIZER = args.optimizer
    #BATCH_SIZE = args.batch_size
    #NB_EPOCH = args.epochs
    
    if not os.path.exists(args.model_fname):
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
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print("Now loading UDC data...")
        
        train_c, train_r, train_l = cPickle.load(open(args.input_dir + 'train.pkl', 'rb'))
        test_c, test_r, test_l = cPickle.load(open(args.input_dir + 'test.pkl', 'rb'))
        dev_c, dev_r, dev_l = cPickle.load(open(args.input_dir + 'dev.pkl', 'rb'))
        
        MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, word_index = cPickle.load(open(args.input_dir + 'params.pkl', 'rb'))
        
        print('Found %s training texts.' % len(train_c)
        print('Found %s dev texts.' % len(dev_c)
        print('Found %s test texts.' % len(test_c)
        
        
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
        branch1 = Sequential()
        branch1.add(Embedding(output_dim=args.emb_dim,
                              input_dim=MAX_NB_WORDS,
                              input_length=MAX_SEQUENCE_LENGTH,
                              weights=[embedding_matrix],
                              mask_zero=True,
                              trainable=True))
        branch1.add(LSTM(output_dim=args.hidden_size))

        # define lstm for sentence2
        branch2 = Sequential()
        branch2.add(Embedding(output_dim=args.emb_dim,
                              input_dim=MAX_NB_WORDS,
                              input_length=MAX_SEQUENCE_LENGTH,
                              weights=[embedding_matrix],
                              mask_zero=True,
                              trainable=True))
        branch2.add(LSTM(output_dim=args.hidden_size))

        # define classifier model
        model = Sequential()
        # Merge layer holds a weight matrix of itself
        model.add(Merge([branch1, branch2], mode='mul'))
        model.add(Dense(1))
        #model.add(Dropout(0.5))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer=args.optimizer)
        
        print(model.summary())
        
        print("Now training the model...")
        
        histories = cnn_callbacks.Histories()
        
        bestAcc = 0.0
        patience = 0 
        
        print("\tbatch_size={}, nb_epoch={}".format(args.batch_size, args.n_epochs))
        
        for ep in range(1, args.n_epochs):
            
            model.fit([train_c, train_r], train_l,
                  batch_size=args.batch_size, nb_epoch=1, callbacks=[histories],
                  validation_data=([dev_c, dev_r], dev_l), verbose=1)

            #model.save(model_name + "_ep." + str(ep) + ".h5")

            curAcc =  histories.accs[0]
            if curAcc >= bestAcc:
                bestAcc = curAcc
                patience = 0
            else:
                patience = patience + 1

            #doing classify the test set
            y_pred = model.predict([test_c, test_r])        
            
            print("Perform on test set after Epoch: " + str(ep) + "...!")    
            #print (y_pred)
            #print (y_pred[:,1], len(y_pred[:,1]))
            recall_k = compute_recall_ks(y_pred[:,0])
            
            #stop the model whch patience = 8
            if patience > 10:
                print("Early stopping at epoch: "+ str(ep))
                break
        
        
        
        if args.save_model:
            print("Now saving the model... at {}".format(args.model_fname))
            model.save(args.model_fname)

    else:
        print("Found pre-trained model...")
        model = K_load_model(args.model_fname)

    return model

if __name__ == "__main__":
    model = load_model()
    