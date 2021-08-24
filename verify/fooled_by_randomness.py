import copy
import glob
import itertools
import numpy as np
import re
import string
import sys
import tensorflow as tf
from pandas import read_csv
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical  # one-hot encode target column


sys.path.append('./../train/')
from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem

# Global parameters
architecture = 'attention'
dataset = 'sst'
maxlen = 15
emb_dims = 50
input_dims = (1, maxlen, emb_dims)
min_accuracy = 0.7  # min accuracy to test a model

# Select the embedding
path_to_embeddings = './../data/embeddings/'
EMBEDDING_FILENAME = path_to_embeddings+'custom-embedding-SST.{}d.txt'.format(emb_dims)
word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)

X_pert = []
for i in range(1, maxlen+1):
    for j in range(1000):
        x = np.zeros(shape=(maxlen*emb_dims))
        x[0:i*emb_dims] += np.random.rand(i*emb_dims)
        X_pert += [x.reshape(input_dims)]

# For each model, test the accuracy on negations
files = glob.glob(f'./../models/{architecture}/{architecture}_{dataset}_inplen-{maxlen}*')
testbed_size = len(X_pert)
negatives, positives = [0 for _ in range(len(files))], [0 for _ in range(len(files))]
partial_neg, partial_pos = [0 for _ in range(len(files))], [0 for _ in range(len(files))]
for exp,f in enumerate(files):
    print(f"Evaluating model {f}...")
    partial_cnt, idx_random_tokens = 0, 0
    model = load_model(f, custom_objects=(SeqSelfAttention.get_custom_objects() if architecture=='attention' else None))
    for i,x in enumerate(X_pert):
        y_hat = model.predict(x)
        if np.argmax(y_hat) == 0:
            negatives[idx_random_tokens] += 1
        else:
            positives[idx_random_tokens] += 1
        partial_cnt += 1
        if partial_cnt == 1000:
            partial_cnt = 0

print(f"\nPercentage of negative lables is {sum(negatives)/testbed_size}.")
print(f"Percentage of positive lables is {sum(positives)/testbed_size}.")
print(f"Distribution of negatives w.r.t. the number of random tokens: {[i/1000 for i in negatives]}")
print(f"Distribution of positives w.r.t. the number of random tokens: {[i/1000 for i in positives]}")