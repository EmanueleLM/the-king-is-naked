import copy as cp
import itertools
import numpy as np
import re
import tensorflow as tf
import torch
import torch.nn as nn
import sys
import tqdm

from collections import deque
from transformers import AdamW, BertTokenizerFast, BertForMaskedLM, BertConfig

from linguistic_augmentation import shallow_negation, mixed_sentiment, sarcasm, name_bias
sys.path.append('./../train/')
from SentimentBERT import SentimentBERT
from text_utils_torch import load_SST, load_IMDB, dataset_to_dataloader

# specify cpu/GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logs only errors
tf.get_logger().setLevel('INFO')

# Training params
dataset = "sst"
load_dataset = (load_IMDB if dataset=="imdb" else load_SST)
maxlen = 15
batch_size = 32

# Load BERT classifier
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased",
                                    output_hidden_states=True)
bert = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)
sentiment_classifier = SentimentBERT(bert, maxlen)
sentiment_classifier.load_state_dict(torch.load('./../models/bert/bert_pretrained_{}_saved_weights_0.90-accuracy.pt'.format(dataset), map_location=device))
sentiment_classifier.eval()
sentiment_classifier.to(device)
optimizer = AdamW(sentiment_classifier.parameters(), lr=3e-5)  
cross_entropy = nn.NLLLoss()

# Load test set
X, Y, replace, label_changing_replacements = name_bias(bert_format=True) 

# generate samples
print("Generating samples...")
X_pert, Y_pert = [], []
interventions, idx_interventions, category_intervention, num_interventions = [], [], [], []
for x,y in tqdm.tqdm(zip(X, Y)):
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
        tmp = cp.copy(x_list)
        for r,i in zip(replacement, idx_interventions[-1]):
            tmp[i] = r
        # Remove spaces
        tmp = re.sub("\s\s+", " ", ' '.join(tmp)).split(' ')
        # Add bert tokens and pad
        tmp = deque(tmp)
        tmp.appendleft('[CLS]')
        if len(tmp) != maxlen:
            tmp.append('[SEP]')
            tmp = list(tmp) + ['pad' for _ in range(maxlen-len(tmp))]
        else:
            tmp[-1] = '[SEP]'
        X_pert += [' '.join(tmp)]
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

# Create the dataloader
test_dataloader = dataset_to_dataloader(X_pert, Y_pert, tokenizer, maxlen, batch_size=batch_size)

# Test the model
print("\nEvaluating...")
sentiment_classifier.eval()
total_loss = 0
total_preds, total_labels = np.array([]), np.array([])
for step,batch in tqdm.tqdm(enumerate(test_dataloader)):
    batch = [t.to(device) for t in batch]
    sent_id, mask, labels = batch
    with torch.no_grad():
        preds = sentiment_classifier(sent_id, mask)
        preds = preds.detach().cpu().numpy()
        total_preds = np.append(total_preds, [np.argmax(p) for p in preds], axis=0)
        total_labels = np.append(total_labels, labels.cpu().detach().numpy(), axis=0)
avg_loss = total_loss / len(test_dataloader) 
accuracies = [1 if p==l else 0 for p,l in zip(total_preds, total_labels)]
print(f"Average accuracy over {len(X_pert)} evaluations: {np.mean(accuracies)} \pm {np.std(accuracies)}")