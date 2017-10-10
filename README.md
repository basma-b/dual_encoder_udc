# dual_encoder_udc
An implementation of dual encoder using Keras

In this repository I implemented the dual encoder used in this paper https://arxiv.org/pdf/1506.08909.pdf using Keras. The motivation of this work is that the available implementations of this work are in Theano https://github.com/npow/ubottu and Tensorflow https://github.com/dennybritz/chatbot-retrieval/ 

I think that both implementations are hard to understand and to re-use (my point of view) so I decided to implement a simple code using Keras.

## Requires:
* Python 2.7
* Theano
* Keras

## Before running the model:
* Creat three folders in the local directory (dual_encoder_udc): dataset, model and embeddings.
* Download Glove Embeddings into embeddings directory 
  ```
  cd embeddings
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip
  unzip glove.840B.300d.zip
  ```
* Download the dataset in format .pkl into dataset folder
  ```
  cd dataset
  wget 
  ```

## Running the model:
```
python dual_encoder.py
```
