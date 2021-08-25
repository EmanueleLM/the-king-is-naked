import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import tqdm

from transformers import AdamW, RobertaTokenizer, RobertaModel, RobertaConfig

from SentimentBERT import SentimentRoberta
from text_utils_torch import load_SST, load_IMDB, dataset_to_dataloader

# function to train the model
def train():
    global model, cross_entropy
    model.train()
    total_loss = 0
    total_preds=[]
    # iterate over batches
    for step,batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        model.zero_grad()        
        preds = model(sent_id, mask)
        loss = cross_entropy(preds, labels)
        total_loss = total_loss + loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds=preds.detach().cpu().numpy()
        total_preds.append(preds)
    avg_loss = total_loss / len(train_dataloader)
    total_preds  = np.concatenate(total_preds, axis=0)
    return avg_loss, total_preds

def evaluate():
    global model, test_dataloader, cross_entropy, device
    print("\nEvaluating...")
    model.eval()
    total_loss = 0
    total_preds, total_labels = np.array([]), np.array([])
    for step,batch in enumerate(test_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))
        batch = [t.to(device) for t in batch]
        sent_id, mask, labels = batch
        with torch.no_grad():
            preds = model(sent_id, mask)
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
maxlen, epochs = 15, 10
batch_size = 64

# Load BERT classifier
tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
config = RobertaConfig.from_pretrained("roberta-large",
                                        output_hidden_states=True)
bert = RobertaModel.from_pretrained('roberta-large', config=config)
# freeze all the bert parameters
for param in bert.parameters():
    param.requires_grad = True
# Add the trainable classification `head`
model = SentimentRoberta(bert, maxlen)

# Load dataset
(X_train, y_train),  (X_test, y_test) = load_dataset(maxlen)

# Create train/test torch dataloaders
train_dataloader = dataset_to_dataloader(X_train, y_train, tokenizer, maxlen, batch_size)
test_dataloader = dataset_to_dataloader(X_test, y_test, tokenizer, maxlen, batch_size)

# Load pre-trained weights, if they exist
try:
    print("Trying to load existing weights from ./../models/language_models/ folder...")
    model.load_state_dict(torch.load('./../models/language_models/roberta-large_pretrained_{}_saved_weights_accuracy-0.84.pt'.format(dataset), map_location=device))
    print("Weights successfully loaded!")
except:
    print(f"No file named roberta-large_pretrained_{dataset}_saved_weights_accuracy-0.84.pt exists under folder ./../models/language_models/")
model = model.to(device)  # push the model to the training device selected
optimizer = AdamW(model.parameters(), lr=3e-5)  
cross_entropy = nn.NLLLoss()

best_valid_loss = float('inf')
train_losses, valid_losses = [], []
valid_preds = []

#for each epoch
for epoch in range(epochs):     
    print('\nEpoch {:} / {:}'.format(epoch + 1, epochs))    
    train_loss, _ = train()    
    valid_loss, total_preds, total_labels = evaluate()
    accuracy = np.mean([1 if p==l else 0 for p,l in zip(total_preds, total_labels)])
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './../models/language_models/roberta-large_pretrained_{}_saved_weights.pt'.format(dataset))    
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_preds.append(total_preds)  
    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')
    print(f'Validation Accuracy: {accuracy:.3f}')
