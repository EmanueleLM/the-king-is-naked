import glob
import numpy as np
import string
import sys
import tensorflow as tf
from keras_self_attention import SeqSelfAttention
from pandas import read_csv
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical  # one-hot encode target column

from models import fc, cnn1d, lstm, attention, import_architecture
from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem
sys.path.append('./../verify/')
from linguistic_augmentation import shallow_negation, sarcasm, mixed_sentiment, name_bias

# Net parameters
maxlen = 25
emb_dims = 50
architecture = 'attention'
test_on_hard_instances = True
input_shape = ((1, maxlen*emb_dims) if architecture=='fc' else (1, maxlen, emb_dims))
init_architecture = import_architecture(architecture)  # import the model template
path_architecture = (architecture if architecture!='rnn' else 'rnn')
custom_object = (SeqSelfAttention.get_custom_objects() if architecture=='attention' else None)
custom_path = ''  # 'augmented_' or ''

# Load trained models
files_ = glob.glob(f"./../models/{path_architecture}/{custom_path}{path_architecture}*")
num_exp = len(files_)  # number of trained networks

# Load test set
X_test, y_test = [], []
accuracies = []
if test_on_hard_instances is False:
    # Load sst test set
    X_test = read_csv('./../data/datasets/SST_2/eval/SST_2__TEST.csv', sep=',',header=None).values
    for i in range(len(X_test)):
        r, s = X_test[i]
        X_test[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
        y_test.append((0 if s.strip()=='negative' else 1))
    X_test = list(X_test[:,0])
else:
    # Load hard instances
    X_hard_train = read_csv('./../data/datasets/sentiment_not_solved/sentiment-not-solved.txt', sep='\t',header=None).values
    for i in range(len(X_hard_train)):
        if X_hard_train[i][1] in ['mpqa', 'opener', 'semeval']:
            r, s = X_hard_train[i][4], int(X_hard_train[i][3])
            if s != 2:
                X_test.append([w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')])
                if s == 0 or s==1:
                    y_test.append(0)
                elif s == 3 or s==4:
                    y_test.append(1)
                else:
                    raise Exception(f"Unexpected value appended to y_test, expected (0,1,3,4), received {s}")
        elif X_hard_train[i][1] in ['tackstrom', 'thelwall']:
            r, s = X_hard_train[i][4], int(X_hard_train[i][3])
            if s != 2:
                X_test.append([w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')])
                if s == 1:
                    y_test.append(0)
                elif s == 3:
                    y_test.append(1)
                else:
                    raise Exception(f"Unexpected value appended to y_test, expected (1,3), received {s}")

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
for exp, m_name in enumerate(files_):
    print(f"\nExperiment {exp+1}/{num_exp}")
    model = load_model(m_name, custom_objects=custom_object)
    _, accuracy = model.evaluate(X_test, y_test, batch_size=64)  # evaluate
    print(f"Raw accuracy of {m_name}: {accuracy}")
    print()
    accuracies += [accuracy]

print(f"Average {architecture} accuracy on `hard instances`: {np.mean(accuracies):.4f} \pm {np.std(accuracies):.4f}")