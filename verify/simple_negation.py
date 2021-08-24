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

# Test 
X = ['this @category@ movie is not @augment@ @positive@',
     'this @category@ movie is not @augment@ @negative@',
     'it is @booltrue@ that this @category@ movie is @augment@ @positive@',
     'it is @boolfalse@ that this @category@ movie is @augment@ @positive@'
    ]
Y = [0, 1, 1, 0]

replace = {}  # first element of each entry is the default and preserve the original label y
replace['@category@'] = ['', 'horror', 'comedy', 'drama', 'thriller']
replace['@augment@'] = ['', 'very', 'incredibly', 'super', 'extremely']
replace['@positive@'] = ['good', 'fantastic', 'nice', 'satisfactory', 'interesting']
replace['@negative@'] = ['bad', 'poor', 'boring', 'terrible', 'awful']
replace['@booltrue@'] = ['true', 'accurate', 'correct', 'right']
replace['@boolfalse@'] = ['false', 'untrue', 'wrong', 'incorrect']
label_changing_replacements = []

# generate samples
X_pert, Y_pert = [], []
interventions, idx_interventions, category_intervention, num_interventions = [], [], [], []
for x,y in zip(X, Y):
    interventions += [[]]
    idx_interventions += [[]]
    category_intervention += [[]]
    x_list = x.split(' ')
    for i,w in enumerate(x_list):
        if w in replace.keys():
            interventions[-1] += [w]
            idx_interventions[-1] += [i]
            category_intervention[-1] += [w]
    res = 0
    for i,c in enumerate(category_intervention[-1]):
        if i == 0:
            res = 1
        res *= len(replace[c])
    num_interventions += [res]
    for replacement in itertools.product(*(replace[r] for r in interventions[-1])):
        tmp = copy.copy(x_list)
        for r,i in zip(replacement, idx_interventions[-1]):
            tmp[i] = r
        X_pert += [tmp]
        # Generate the label, knowing that an intervention in the list
        #  label_changing_replacements changes the original label y iff it
        #  imposes a replacement whose index is strictly greater that 0.
        y_tmp = y
        for c,i in zip(category_intervention[-1], idx_interventions[-1]):
            if c not in label_changing_replacements:
                continue
            else:
                flip = lambda v: 1 if v==0 else 0
                y_tmp = (flip(y_tmp) if replace[c].index(tmp[i])>0 else y_tmp)
        Y_pert += [y_tmp]

# Select the embedding
path_to_embeddings = './../data/embeddings/'
EMBEDDING_FILENAME = path_to_embeddings+'custom-embedding-SST.{}d.txt'.format(emb_dims)
word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)

# For each model, test the accuracy on negations
total_correct, accuracies = 0, []
testbed_size = len(X_pert)
files = glob.glob(f'./../models/{architecture}/{architecture}_{dataset}_inplen-{maxlen}*')
for exp,f in enumerate(files):
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