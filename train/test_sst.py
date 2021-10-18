import glob
import numpy as np
import string
import sys
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
from pandas import read_csv
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical  # one-hot encode target column

from models import fc, cnn1d, cnn2d, lstm, attention, import_architecture
from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem
sys.path.append('./../verify/')
from linguistic_augmentation import shallow_negation, sarcasm, mixed_sentiment, name_bias

# Net parameters
maxlen = 25
emb_dims = 50
architecture = 'cnn2d'
test_type = 'linguistic_phenomena'  # 'sst', 'linguistic_phenomena', 'sentiment_not_solved' 
test_augmented_networks = True
test_IBP = False  # IBP path
test_rule = shallow_negation  # a linguistic_augmentation function
if test_rule.__name__ == 'shallow_negation':
    r1 = r2 = 'negated'
elif test_rule.__name__ == 'mixed_sentiment':
    r1 = r2 = 'mixed'
elif test_rule.__name__ == 'sarcasm':
    r1, r2 = 'sarcasm', 'irony'
else:
    raise Exception(f"{test_rule.__name__} is not a valid test_rule (shallow_negation, mixed_sentiment, sarcasm)")
    
# Input shape
if architecture == 'fc':
    input_shape = (1, maxlen*emb_dims)    
elif architecture == 'cnn2d':
    if int(maxlen**0.5 * maxlen**0.5) != maxlen:
        raise Exception(f"{maxlen} is not a square number.")
    k_size = int(maxlen**0.5)
    input_shape = (1, k_size, k_size, emb_dims)
else:
    input_shape = (1, maxlen, emb_dims)

init_architecture = import_architecture(architecture)  # import the model template
path_architecture = (architecture if architecture!='rnn' else 'rnn')
custom_object = (SeqSelfAttention.get_custom_objects() if architecture=='attention' else None)
custom_path = ('vanilla' if test_augmented_networks is False else str(test_rule.__name__))  # 'augmented_' or ''

# Load trained models
if test_IBP is False:
    files_ = glob.glob(f"./../models/{path_architecture}/{custom_path}/*{path_architecture}*")
else:
    files_ = glob.glob(f"./../models/IBP/{path_architecture}/{custom_path}/*{path_architecture}*")

num_exp = len(files_)  # number of trained networks

# Load test set
X_test, y_test = [], []
accuracies = []
if test_type == 'sst':
    # Load sst test set
    X_test = read_csv('./../data/datasets/SST_2/eval/SST_2__TEST.csv', sep=',',header=None).values
    for i in range(len(X_test)):
        r, s = X_test[i]
        X_test[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
        y_test.append((0 if s.strip()=='negative' else 1))
    X_test = list(X_test[:,0])
elif test_type == 'sentiment_not_solved':
    # Load hard instances
    X_hard_train = read_csv('./../data/datasets/sentiment_not_solved/sentiment-not-solved.txt', sep='\t',header=None).values
    for i in range(len(X_hard_train)):
        if X_hard_train[i][1] in ['sst', 'mpqa', 'opener', 'semeval'] and (r1 in X_hard_train[i][-1] or r2 in X_hard_train[i][-1]):
            r, s = X_hard_train[i][4], int(X_hard_train[i][3])
            if s != 2:
                X_test.append([w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')])
                if s == 0 or s==1:
                    y_test.append(0)
                elif s == 3 or s==4:
                    y_test.append(1)
                else:
                    raise Exception(f"Unexpected value appended to y_test, expected (0,1,3,4), received {s}")
        elif X_hard_train[i][1] in ['sst', 'tackstrom', 'thelwall'] and (r1 in X_hard_train[i][-1] or r2 in X_hard_train[i][-1]):
            r, s = X_hard_train[i][4], int(X_hard_train[i][3])
            if s != 2:
                X_test.append([w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')])
                if s == 1:
                    y_test.append(0)
                elif s == 3:
                    y_test.append(1)
                else:
                    raise Exception(f"Unexpected value appended to y_test, expected (1,3), received {s}")
else:
    # Augment with all the linguistic rules
    for l_rule in [test_rule]:
        X_augment, y_augment = l_rule()
        X_test = X_test+X_augment
        y_test = y_test+y_augment

# Select the embedding
embedding_name = 'custom-embedding-SST.{}d.txt'.format(emb_dims)
path_to_embeddings = './../data/embeddings/'
EMBEDDING_FILENAME = path_to_embeddings+embedding_name
print(f"Loading {embedding_name} embedding...")
word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)

# Prepare test set 
X_test = np.array([np.array(x) for x in X_test]) 
X_test = [[index2embedding[word2index[x]] for x in xx] for xx in X_test]
X_test = np.asarray(pad_sequences(X_test, maxlen=maxlen, emb_size=emb_dims))
X_test = X_test.reshape(len(X_test), *input_shape[1:])
y_test = to_categorical(y_test, num_classes=2)

# Train and save the models
accuracies = []
W_means = []
for exp, m_name in enumerate(files_):
    print(f"\nExperiment {exp+1}/{num_exp}")
    model = load_model(m_name, custom_objects=custom_object)
    _, accuracy = model.evaluate(X_test, y_test, batch_size=64)  # evaluate
    W = []
    for w in model.weights:
        W += [w.numpy().flatten().tolist()]
    W = [item for sublist in W for item in sublist]
    W_means += [np.mean(W)]
    print(f"Raw accuracy of {m_name}: {accuracy}")
    print()
    accuracies += [accuracy]

print(f"Average {architecture} accuracy on `{test_type}`: {np.mean(accuracies):.4f} \pm {np.std(accuracies):.4f}")
print(f"Weights norm: {np.mean(W_means):.4f} \pm {np.std(W_means):.4f}")