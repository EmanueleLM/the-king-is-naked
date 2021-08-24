import copy as cp
import gensim.downloader as api
import keras
import nltk
import numpy as np
import random
import torch

from functools import lru_cache
#from gensim.models import Word2Vec
from nltk.corpus import wordnet
from pandas import read_csv
from scipy import spatial
from sklearn.decomposition import PCA as pca_analysis
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

EMBEDDING_NAME = 'word2vec-google-news-300'

@lru_cache(maxsize=None)
def import_Word2VecModel(embedding_name):
    print("Importing text_utils module. The first time it can take a while...")
    w2v = api.load('word2vec-google-news-300')
    return w2v

def get_difference(l1, l2):
    """
    Return the word to word difference between two lists
    """
    cnt, diff = 0, []
    maxlen = min(len(l1), len(l2))
    for i, el1, el2 in zip(range(maxlen), l1[:maxlen], l2[:maxlen]):
        if el1 != el2:
            cnt += 1
            diff += [i]
    return cnt, diff

def random_combination(iterable, r, sims):
    i = 0
    pool = tuple(iterable)
    n = len(pool)
    rng = range(n)
    while i < sims:
        i += 1
        rr = random.randint(1, r)
        yield [pool[j] for j in random.sample(rng, rr)]

def word_mover_distance(s1, s2):
    """
    Default word2vec is 'word2vec-google-news-300' from gensim
    """
    Word2VecModel = import_Word2VecModel(EMBEDDING_NAME)
    return Word2VecModel.wmdistance(s1, s2)

def load_SST(maxlen):
    """
    maxlen is the number of words in each input text (eventually padded/cut)
    output:
     list of lists with train and test inputs (strings) and labels (0, 1).
    """
    # Load STT dataset (Tokenize)
    X_train = read_csv('./../data/datasets/SST_2/training/SST_2__FULL.csv', sep=',',header=None).values
    X_test = read_csv('./../data/datasets/SST_2/eval/SST_2__TEST.csv', sep=',',header=None).values
    y_train, y_test = [], []
    for i in range(len(X_train)):
        r, s = X_train[i]  # review, score (comma separated in the original file)
        X_train[i][0] = r.split(' ')[:maxlen-2]
        X_train[i][0] += ['[SEP]']; X_train[i][0].insert(0, '[CLS]')
        X_train[i][0] += ['pad']*(maxlen-len(X_train[i][0]))
        X_train[i][0] = ' '.join(X_train[i][0])
        y_train.append((0 if s.strip()=='negative' else 1))
    for i in range(len(X_test)):
        r, s = X_test[i]
        X_test[i][0] = r.split(' ')[:maxlen-2]
        X_test[i][0] += ['[SEP]']; X_test[i][0].insert(0, '[CLS]')
        X_test[i][0] += ['pad']*(maxlen-len(X_test[i][0]))
        X_test[i][0] = ' '.join(X_test[i][0])
        y_test.append((0 if s.strip()=='negative' else 1))
    X_train, X_test = X_train[:,0].tolist(), X_test[:,0].tolist()
    return (X_train, y_train), (X_test, y_test)

def load_IMDB(maxlen):
    """
    maxlen is the number of words in each input text (eventually padded/cut)
    output:
     list of lists with train and test inputs (strings) and labels (0, 1).
    """
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=5000, index_from=2)
    word_to_id = keras.datasets.imdb.get_word_index()
    word_to_id = {k:(v+2) for k,v in word_to_id.items()}
    word_to_id["pad"] = 0
    word_to_id["[CLS]"] = 1
    word_to_id["unk"] = 2
    id_to_word = {value:key for key,value in word_to_id.items()}
    X_train, X_test = [' '.join([id_to_word[w] for w in x]) for x in X_train], [' '.join([id_to_word[w] for w in x]) for x in X_test]
    for i,_ in enumerate(X_train):
        X_train[i] = X_train[i].split(' ')[:maxlen-1]
        X_train[i] +=  ['pad']*(maxlen-len(X_train[i]))
        X_train[i][-1] = '[SEP]'
        X_train[i] = ' '.join(X_train[i])
    for i,_ in enumerate(X_test):
        X_test[i] = X_test[i].split(' ')[:maxlen-1]
        X_test[i] +=  ['pad']*(maxlen-len(X_test[i]))
        X_test[i][-1] = '[SEP]'
        X_test[i] = ' '.join(X_test[i])
    return (X_train, y_train), (X_test, y_test)

def dataset_to_dataloader(X, y, tokenizer, maxlen, batch_size=32, device=torch.device("cpu")):
    """
    X, y are the outputs of load_SST or load_IMDB
    tokenizer is a standard (bert) tokenizer
    maxlen is the number of words in each input text (eventually padded/cut)
    """ 
    X_tokens = tokenizer.batch_encode_plus(
        X,
        max_length = maxlen,
        pad_to_max_length=False,
        truncation=False
    )
    X_seq = torch.tensor(X_tokens['input_ids'])
    X_mask = torch.tensor(X_tokens['attention_mask'])
    y = torch.tensor(y)
    # Create dataloaders
    X_data = TensorDataset(X_seq, X_mask, y)
    X_sampler = RandomSampler(X_data)
    X_dataloader = DataLoader(X_data, sampler=X_sampler, batch_size=batch_size)
    return X_dataloader

def text_to_concepts(text, concepts, word2vec, embedding_dims, pca_components=5, selection='max'):
    """
    text:list
     input text as a list of words (e.g., ['sarah', 'liked', 'the', 'horror', 'movie']).
    concepts:dictionary
     key is the name of each concept, value is a list of words associated to that concept 
     (e.g., concepts['genre']=['horror', 'comedy', 'action']).
    word2vec:function
     expects a function with a single argument that takes as input a word (string) and returns a
     vector (np.array). The function should return a value for any string input (i.e., KeyError 
     must be handled).
    embedding_dims:int
     dimension of each word embedding
    pca_components:int
     (optional) the number of dimensions retained by the PCA to identify a concept subspace (try 5, 10, maybe more).
    selection:str
     (optional) set to `max` to assign a word to the subspace that contains the term that minimizes the PCA
      likelihood. `avg` averages the likelihoods for each term in the concept.
    output:
     assignments: list
      for each element in text, assignments contains the name of the concept it has been assigned to.
    TODO: to fasten segmentation of a text (e.g., when doing lot of subs), implement a sampling-based technique when
     cyclying on text_w2v (or equivalently, vecotrize stuff).
    TODO: implement a cosine similarity filter on couples of synonyms that shouldn't be used to identify a subspace.
    """
    PCA = pca_analysis(n_components=pca_components)
    concepts_vectors, pca = {k:v for k,v in zip(concepts.keys(), [[] for _ in range(len(concepts))])}, {}
    for key in concepts.keys():                
        #print("Processing key {}".format(key))
        for w1 in concepts[key]:
            for w2 in concepts[key]:
                if w1 != w2:
                    e1, e2 = word2vec(w1).reshape(embedding_dims), word2vec(w2).reshape(embedding_dims) 
                    #print(f"d({w1}, {w2})", spatial.distance.cosine(e1, e2))
                    #if spatial.distance.cosine(e1, e2) > 0.:  # words should be at least correlated
                    concepts_vectors[key] += [e1 - e2]
        pca[key] = cp.copy(PCA.fit(np.array(concepts_vectors[key])))
    assignments, text_w2v = [], []
    from_now_on = -1  # point where [SEP] is inserted (every word after is assigned to none concept)
    for i,w in enumerate(text):
        if w == '[SEP]':
            from_now_on = i
        text_w2v += [word2vec(w).reshape(1, embedding_dims)]
    concepts_list, likelihoods = list(concepts.keys()), []
    for i,e in enumerate(text_w2v):
        likelihood = []
        if i >= from_now_on and from_now_on != -1:
            likelihoods += [[(0. if c!='none' else 1.) for c in concepts]]
            likelihoods[-1][-1] = 1.
        for key in concepts:
            if selection == 'max':
                max_ = -np.inf
                for w in concepts[key]:
                    #print(f"d({text[i]}, {w})", spatial.distance.cosine(e, word2vec(w).reshape(embedding_dims)))
                    #if spatial.distance.cosine(e, word2vec(w).reshape(embedding_dims)) > 0.:  # words should be at least correlated
                    d = e - word2vec(w).reshape(embedding_dims)
                    if pca[key].score(d) > max_:
                        max_ = pca[key].score(d)
                likelihood += [max_]
            elif selection == 'avg':
                avg, cnt = 0., 0
                for w in concepts[key]:
                    #if spatial.distance.cosine(e, word2vec(w).reshape(embedding_dims)) > 0.:  # words should be at least correlated
                    d = e - word2vec(w).reshape(embedding_dims)
                    avg += pca[key].score(d)
                    #cnt += 1
                likelihood += [avg/cnt]
            else:
                raise NotImplementedError(f"{selection} is not a valid `selection` method. Use `max`, or `avg`.")
        likelihoods += [likelihood]        
        #print(likelihoods[-1])
        #v = np.max(likelihoods[-1]) 
        l = np.argmax(likelihoods[-1])
        assignments.append(concepts_list[l])
    #print(likelihoods)
    return assignments

def text_to_concepts_bert(text, concepts, word2vec, embedding_dims, pca_components=5, selection='max'):
    """
    text:list
     input text as a list of words (e.g., ['sarah', 'liked', 'the', 'horror', 'movie']).
    concepts:dictionary
     key is the name of each concept, value is a list of words associated to that concept 
     (e.g., concepts['genre']=['horror', 'comedy', 'action']).
    word2vec:function
     expects a function with two arguments, the index of the word from a bert-compatible 
      string (i.e., tokenized) and the string itself. It returns a vector (np.array). 
      The function should return a value for any string input (i.e., KeyError must be handled).
    embedding_dims:int
     dimension of each word embedding
    pca_components:int
     (optional) the number of dimensions retained by the PCA to identify a concept subspace (try 5, 10, maybe more).
    selection:str
     (optional) set to `max` to assign a word to the subspace that contains the term that minimizes the PCA
      likelihood. `avg` averages the likelihoods for each term in the concept.
    output:
     assignments: list
      for each element in text, assignments contains the name of the concept it has been assigned to.
    TODO: to fasten segmentation of a text (e.g., when doing lot of subs), implement a sampling-based technique when
     cyclying on text_w2v (or equivalently, vecotrize stuff).
    TODO: implement a cosine similarity filter on couples of synonyms that shouldn't be used to identify a subspace.
    """
    PCA = pca_analysis(n_components=pca_components)
    concepts_vectors, pca = {k:v for k,v in zip(concepts.keys(), [[] for _ in range(len(concepts))])}, {}
    for key in concepts.keys():                
        #print("Processing key {}".format(key))
        for i,w1 in enumerate(concepts[key]):
            for w2 in concepts[key]:
                if w1 != w2:
                    e1, e2 = word2vec(-4, "[CLS] The category of word {} is {} [SEP]".format(w1, key).split(' ')).reshape(embedding_dims), word2vec(-4, "[CLS] The category of word {} is {} [SEP]".format(w2, key).split(' ')).reshape(embedding_dims)
                    concepts_vectors[key] += [e1 - e2]
        pca[key] = cp.copy(PCA.fit(np.array(concepts_vectors[key])))
    assignments, text_w2v = [], []
    # contextualized bert embedding
    for i,w in enumerate(text):
        text_w2v += [word2vec(i, text).reshape(1, embedding_dims)]
    concepts_list, likelihoods = list(concepts.keys()), []
    for i,e in enumerate(text_w2v):
        likelihood = []
        for key in concepts:
            if selection == 'max':
                max_ = -np.inf
                for w in concepts[key]:
                    d = e - word2vec(-4, "[CLS] The category of word {} is {} [SEP]".format(w, key).split(' ')).reshape(embedding_dims)
                    if pca[key].score(d) > max_:
                        max_ = pca[key].score(d)
                likelihood += [max_]
            elif selection == 'avg':
                avg, cnt = 0., 0
                for w in concepts[key]:
                    d = e - word2vec(-4, "[CLS] The category of word {} is {} [SEP]".format(w, key).split(' ')).reshape(embedding_dims)
                    avg += pca[key].score(d)
                likelihood += [avg/cnt]
            else:
                raise NotImplementedError(f"{selection} is not a valid `selection` method. Use `max`, or `avg`.")
        likelihoods += [likelihood]        
        #print(likelihoods[-1])
        #v = np.max(likelihoods[-1]) 
        l = np.argmax(likelihoods[-1])
        assignments.append(concepts_list[l])
    #print(likelihoods)
    return assignments
