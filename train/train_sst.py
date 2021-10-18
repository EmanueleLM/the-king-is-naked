import numpy as np
import string
import sys
import tensorflow as tf
from pandas import read_csv
from tensorflow.keras.utils import to_categorical  # one-hot encode target column

from models import fc, cnn1d, cnn2d, lstm, attention, import_architecture
from glove_utils import load_embedding, pad_sequences
from text_utils import clean_text, stem_lem
sys.path.append('./../verify/')
from linguistic_augmentation import shallow_negation, sarcasm, mixed_sentiment, name_bias

# Net parameters
maxlen = 25
emb_dims = 50
epochs = 20
num_exp = 10  # number of trained networks
finetune_on_hard_instances = True
architecture = 'cnn2d'
data_augmentation = 500  # multiplicative factor for further training data

# test rules and custom path
augment_rule1 = 'negated'
augment_rule2 = 'negated'
if finetune_on_hard_instances is False:
    custom_path = 'vanilla'
elif augment_rule1 == 'negated':
    custom_path = 'shallow_negation'
elif augment_rule1 == 'mixed':
    custom_path = 'mixed_sentiment'
elif augment_rule1 == 'irony' or augment_rule1 == 'sarcasm':
    custom_path = 'sarcasm'
else:
    pass  # this leaves custom_path undefined --> raise error

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

# Import architecture 
init_architecture = import_architecture(architecture)  # import the model template

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
X_train, X_test = list(X_train[:,0]), list(X_test[:,0])

# Setup hard instances
if finetune_on_hard_instances is True:
    # Fine-tune on the the sst-examples from the `sentiment-is-not-solved` dataset
    X_hard_train = read_csv('./../data/datasets/sentiment_not_solved/sentiment-not-solved.txt', sep='\t',header=None).values
    X_augment, y_augment = [], []
    for i in range(len(X_hard_train)):
        if X_hard_train[i][1] in ['sst', 'mpqa', 'opener', 'semeval'] and (augment_rule1 in X_hard_train[i][-1] or augment_rule2 in X_hard_train[i][-1]):
            r, s = X_hard_train[i][4], int(X_hard_train[i][3])
            if s != 2:
                for _ in range(data_augmentation):
                    X_augment.append([w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')])
                if s == 0 or s==1:
                    for _ in range(data_augmentation):
                        y_augment.append(0)
                elif s == 3 or s==4:
                    for _ in range(data_augmentation):
                        y_augment.append(1)
                else:
                    raise Exception(f"Unexpected value appended to y_augment, expected (0,1,3,4), received {s}")
        elif X_hard_train[i][1] in ['tackstrom', 'thelwall'] and (augment_rule1 in X_hard_train[i][-1] or augment_rule2 in X_hard_train[i][-1]):
            r, s = X_hard_train[i][4], int(X_hard_train[i][3])
            if s != 2:
                for _ in range(data_augmentation):
                    X_augment.append([w.lower() for w in r.translate(str.maketrans('', '', string.punctuation)).strip().split(' ')])
                if s == 1:
                    for _ in range(data_augmentation):
                        y_augment.append(0)
                elif s == 3:
                    for _ in range(data_augmentation):
                        y_augment.append(1)
                else:
                    raise Exception(f"Unexpected value appended to y_augment, expected (1,3), received {s}")
    # Augment with hard instances
    X_train = X_train+X_augment
    y_train = y_train+y_augment
    """
    # Augment with all the linguistic rules
    for l_rule in [mixed_sentiment]:
        X_augment, y_augment = l_rule()
        X_train = X_train+X_augment
        y_train = y_train+y_augment
    """

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
    old_accuracy = -np.inf
    print(f"\nExperiment {exp}/{num_exp}")
    model = init_architecture(input_shape)
    for e in range(epochs):
        for size in range(0, len(X_train), chunk_size):
            X_train_chunk = [[index2embedding[word2index[x]] for x in xx] for xx in X_train[size: size+chunk_size]]        
            X_train_chunk = np.asarray(pad_sequences(X_train_chunk, maxlen=maxlen, emb_size=emb_dims))
            X_train_chunk = X_train_chunk.reshape(len(X_train_chunk), *input_shape[1:])
            y_train_chunk = to_categorical(y_train[size: size+chunk_size], num_classes=2)
            model.fit(X_train_chunk, y_train_chunk, batch_size=256, epochs=1)
        _, accuracy = model.evaluate(X_test, y_test, batch_size=64)  # evaluate
        # Early stopping
        #if accuracy < old_accuracy:
        #    break
        #old_accuracy = accuracy

    # Save trained model
    accuracies += [accuracy]
    print(f"Saving model with accuracy {accuracy}\n")
    model.save(f"./../models/{architecture}/{custom_path}/{architecture}_sst_inplen-{maxlen}_emb-sst{emb_dims}d_exp-{exp}_acc-{accuracy:.6f}") 

print(f"Average {architecture} accuracy: {np.mean(accuracies)} \pm {np.std(accuracies)}")