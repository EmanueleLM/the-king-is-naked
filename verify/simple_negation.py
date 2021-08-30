import copy
import glob
import itertools
import numpy as np
import re
import string
import sys
import tensorflow as tf
import tqdm
from pandas import read_csv
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical  # one-hot encode target column

from linguistic_augmentation import shallow_negation
sys.path.append('./../train/')
from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem

# Global parameters
architecture = 'attention'
dataset = 'sst'
maxlen = 15
emb_dims = 50
input_dims = ((1, maxlen*emb_dims) if architecture=='fc' else (1, maxlen, emb_dims))
custom_path = ''  # 'augmented_' or ''

# Load test set
X_pert, Y_pert = shallow_negation()

# Select the embedding
path_to_embeddings = './../data/embeddings/'
EMBEDDING_FILENAME = path_to_embeddings+'custom-embedding-SST.{}d.txt'.format(emb_dims)
word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)

# For each model, test the accuracy on negations
total_correct, accuracies = 0, []
testbed_size = len(X_pert)
files = glob.glob(f'./../models/{architecture}/{custom_path}{architecture}_{dataset}_inplen-{maxlen}*')
for exp,f in tqdm.tqdm(enumerate(files)):
    partial_correct = 0
    model = load_model(f, custom_objects=(SeqSelfAttention.get_custom_objects() if architecture=='attention' else None))
    for x,y in zip(X_pert, Y_pert):
        x = ' '.join(x)
        x = re.sub("\s\s+", " ", x).split(' ')  # remove multiple spaces (e.g., label-preserving subs)
        x = [index2embedding[word2index[w]] for w in x]
        x = np.asarray(pad_sequences([x], maxlen=maxlen, emb_size=emb_dims)).reshape(*input_dims)
        y_hat = model.predict(x)        
        if np.argmax(y_hat) == y:
            partial_correct += 1
            total_correct += 1
    accuracies += [partial_correct/testbed_size]
    print(f"Partial accuracy of model {f}, exp {exp+1}/{len(files)}: {partial_correct/testbed_size}")
print(f"Average accuracy over {len(files)*testbed_size} evaluations: {np.mean(accuracies)} \pm {np.std(accuracies)}")