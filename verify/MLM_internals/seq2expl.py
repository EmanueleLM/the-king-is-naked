import copy as cp
import itertools
import numpy as np
import re
import string
import tensorflow as tf
import torch
import torch.nn as nn
import sys
import tqdm

from collections import deque
from pandas import read_csv
from transformers import AdamW, BertTokenizerFast, BertForMaskedLM, BertConfig

from masked_interventions import sequential_interventions, most_likely_token_bottom_up
sys.path.append('./../../train/')
from SentimentBERT import SentimentBERT
from text_utils_torch import dataset_to_dataloader

device = 'cpu'

def same_list(a,b):
    for aa,bb in zip(a[1:-1],b[1:-1]):
        if aa!=bb:
            return False
    return True

# Logs only errors
tf.get_logger().setLevel('INFO')

# Load the tokenizer and bert config
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased",
                                    output_hidden_states=True)
bert = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)

"""
# Load pre-trained weights
maxlen = 50
bert = SentimentBERT(bert, maxlen)
bert.load_state_dict(torch.load('./../../models/language_models/bert_pretrained_sst_saved_weights.pt', map_location=device))
"""

for param in bert.parameters():
    param.requires_grad = False

#r = sequential_interventions(bert, input_text, interventions, topk=10, combine_subs=False, mask="[MASK]", budget=100, device=device, TOKENIZER=tokenizer)

input_text = "[CLS] there is no doubt that this movie is bad , actors are incredibly awful [SEP]".split(" ")
masked_text = ["[MASK]" for _ in input_text]
chain = []
while not(same_list(input_text, masked_text)):    
    r = most_likely_token_bottom_up(bert, masked_text, input_text, device=device, TOKENIZER=tokenizer)
    chain += [input_text[r]]
    #input_text[r] = '[MASK]'
    masked_text[r] = input_text[r]

print(chain)