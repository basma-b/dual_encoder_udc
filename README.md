# dual_encoder_udc
An implementation of dual encoder using Keras

In this repository I implemented the dual encoder used in this paper https://arxiv.org/pdf/1506.08909.pdf using Keras. The motivation of this work is that the available implementations of this work are in Theano https://github.com/npow/ubottu and Tensorflow https://github.com/dennybritz/chatbot-retrieval/ 

I think that both implementations are hard to understand and to re-use (my point of view) so I decided to implement a simple code using Keras with Theano in backend.

## Requires:
* Python 3.5
* Theano 0.9.0
* Keras 2.0.8

## Before running the model:
* Creat three folders in the local directory (dual_encoder_udc): dataset, model and embeddings.
* Download Glove Embeddings into embeddings directory 
  ```
  cd embeddings
  wget http://nlp.stanford.edu/data/glove.840B.300d.zip
  unzip glove.840B.300d.zip
  ```
* Download the dataset in format .pkl into dataset folder: This pkl files were generated using the utilities/prepare_dataset.py script which separate the context and response from each of the train.csv, test.csv and dev.csv taht you can download from here https://github.com/rkadlec/ubuntu-ranking-dataset-creator by running ./generate.sh without any options. In case you need this raw data, just email me I can provide them upon request.

  ```
  https://drive.google.com/file/d/1VjVzY0MqKj0b-q_KXnaHC49qCw3iDqEY/view?usp=sharing
  ```

## Running the model:
```
python3.5 dual_encoder.py
```
## Results
Running The above python script should give you the results below. Note that the best results are obtained after epoch 2, using the default hyperparameters: --emb_dim=300, --hidden_size=300, --batch_size=256. Note that embeddings are intilized with glove and are fine tuned during training.

```
Perform on test set after Epoch: 2...!

group_size: 2
recall@1 0.898414376321353
group_size: 5
recall@1 0.7420190274841437
recall@2 0.898784355179704
group_size: 10
recall@1 0.6213002114164905
recall@2 0.7822410147991543
recall@5 0.948784355179704
```
