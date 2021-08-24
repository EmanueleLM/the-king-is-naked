import numpy as np
import string
import tensorflow as tf
from pandas import read_csv
from tensorflow.keras.utils import to_categorical  # one-hot encode target column

from models import fc, cnn1d, rnn, attention, import_architecture
from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem

# Net parameters
maxlen = 15
emb_dims = 50
epochs = 5
num_exp = 30  # number of trained networks
architecture = 'attention'
input_shape = (1, maxlen, emb_dims)
init_architecture = import_architecture(architecture)  # import the model template
model = init_architecture(input_shape)

# Load STT dataset (eliminate punctuation, add padding etc.)
print("[logger]: building STT custom model with tf version {}".format(tf.__version__))
X_train = read_csv('./../data/datasets/SST_2/training/SST_2__FULL.csv', sep=',',header=None).values
X_test = read_csv('./../data/datasets/SST_2/eval/SST_2__TEST.csv', sep=',',header=None).values
y_train, y_test = [], []
for i in range(len(X_train)):
    r, s = X_train[i]  # review, score (comma separated in the original file)
    X_train[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
    y_train.append((0 if s.strip()=='negative' else 1))
for i in range(len(X_test)):
    r, s = X_test[i]
    X_test[i][0] = [w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')]
    y_test.append((0 if s.strip()=='negative' else 1))
X_train, X_test = X_train[:,0], X_test[:,0]

# Select the embedding
embedding_name = 'custom-embedding-SST.{}d.txt'.format(emb_dims)
path_to_embeddings = './../data/embeddings/'
EMBEDDING_FILENAME = path_to_embeddings+embedding_name
print(f"Loading {embedding_name} embedding...")
word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)

# Inputs as Numpy arrays
X_train = np.array([np.array(x) for x in X_train])
X_test = np.array([np.array(x) for x in X_test]) 

# Split dataset into chunks to prevent memory errors
chunk_size = 50000
# Prepare test set in advance
X_test = [[index2embedding[word2index[x]] for x in xx] for xx in X_test]
X_test = np.asarray(pad_sequences(X_test, maxlen=maxlen, emb_size=emb_dims))
X_test = X_test.reshape(len(X_test), *input_shape[1:])
y_test = to_categorical(y_test, num_classes=2)

# Train and save the models
accuracies = []
for exp in range(num_exp):
    print(f"\nExperiment {exp}/{num_exp}")
    for e in range(epochs):
        for size in range(0, len(X_train), chunk_size):
            X_train_chunk = [[index2embedding[word2index[x]] for x in xx] for xx in X_train[size: size+chunk_size]]        
            X_train_chunk = np.asarray(pad_sequences(X_train_chunk, maxlen=maxlen, emb_size=emb_dims))
            X_train_chunk = X_train_chunk.reshape(len(X_train_chunk), *input_shape[1:])
            y_train_chunk = to_categorical(y_train[size: size+chunk_size], num_classes=2)
            model.fit(X_train_chunk, y_train_chunk, batch_size=64, epochs=1)
        _, accuracy = model.evaluate(X_test, y_test, batch_size=64)  # evaluate

    # Save trained model
    accuracies += [accuracy]
    print(f"Saving model with accuracy {accuracy}\n")
    model.save(f"./../models/{architecture}/{architecture}_sst_inplen-{maxlen}_emb-sst{emb_dims}d_exp-{exp}_acc-{accuracy:.6f}") 

print(f"Average {architecture} accuracy: {np.mean(accuracies)} \pm {np.std(accuracies)}")