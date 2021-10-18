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

from linguistic_augmentation import shallow_negation, mixed_sentiment, sarcasm, name_bias
sys.path.append('./../train/')
from SentimentBERT import SentimentBERT
from text_utils_torch import load_SST, load_IMDB, dataset_to_dataloader

# function to train the model
def train():
    global sentiment_classifier, cross_entropy
    sentiment_classifier.train()
    total_loss = 0
    total_preds=[]
    # iterate over batches
    for step,batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        sentiment_classifier.zero_grad()        
        preds = sentiment_classifier(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sentiment_classifier.parameters(), 1.0)
        optimizer.step()
        preds=preds.detach().cpu().numpy()
        total_preds.append(preds)
    avg_loss = total_loss / len(train_dataloader)
    total_preds  = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

def evaluate():
    global sentiment_classifier, test_dataloader, cross_entropy, device
    print("\nEvaluating...")
    sentiment_classifier.eval()
    total_loss = 0
    total_preds, total_labels = np.array([]), np.array([])
    for step,batch in enumerate(test_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            preds = sentiment_classifier(sent_id, mask)
            loss = cross_entropy(preds,labels)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds = np.append(total_preds, [np.argmax(p) for p in preds], axis=0)
            total_labels = np.append(total_labels, labels.cpu().detach().numpy(), axis=0)
    avg_loss = total_loss / len(test_dataloader) 
    return avg_loss, total_preds, total_labels

# specify cpu/GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logs only errors
tf.get_logger().setLevel('INFO')

# Training params
dataset = "sst"
load_dataset = (load_IMDB if dataset=="imdb" else load_SST)
maxlen = 50
batch_size = 32
epochs = 10
num_layers = 16

# Load BERT classifier
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
config = BertConfig.from_pretrained("bert-base-uncased",
                                    output_hidden_states=True)
bert = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)

# freeze all the bert parameters
for param in bert.parameters():
    param.requires_grad = False

# Add the trainable classification `head`
sentiment_classifier = SentimentBERT(bert, maxlen)
#sentiment_classifier.load_state_dict(torch.load('./../models/bert/bert_pretrained_{}_saved_weights_0.90-accuracy.pt'.format(dataset), map_location=device))
sentiment_classifier.eval()
sentiment_classifier.to(device)
optimizer = AdamW(sentiment_classifier.parameters(), lr=3e-5)  
cross_entropy = nn.NLLLoss()

# Load dataset
(X_train, y_train),  (X_test, y_test) = load_dataset(maxlen)

# Create train/test torch dataloaders
train_dataloader = dataset_to_dataloader(X_train, y_train, tokenizer, maxlen, batch_size)
test_dataloader = dataset_to_dataloader(X_test, y_test, tokenizer, maxlen, batch_size)

best_valid_loss = float('inf')
train_losses, valid_losses = [], []
valid_preds = []

# Train the model
for epoch in range(epochs):     
    print('\nEpoch {:} / {:}'.format(epoch + 1, epochs))    
    train_loss, _ = train()    
    valid_loss, total_preds, total_labels = evaluate()
    accuracy = np.mean([1 if p==l else 0 for p,l in zip(total_preds, total_labels)])
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(sentiment_classifier.state_dict(), './../models/language_models/ablation/bert-large_pretrained_{}_saved_weights_num_layers-{}.pt'.format(dataset, num_layers))    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_preds.append(total_preds)  
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')
    print(f'Validation Accuracy: {accuracy:.3f}')
