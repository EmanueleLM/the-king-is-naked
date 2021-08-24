import numpy as np
import torch
import torch.nn as nn

class SentimentBERT(nn.Module):
    """
    This is thought as a sentiment analysis classifier `head` for
     the transformers.BertForMaskedLM module. An example of usage is 
     provided in finetune_bert_sst.py.
    """
    def __init__(self, bert, maxlen):      
        super(SentimentBERT, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768*maxlen, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten(1, -1)

    # Need to return only the softmax output
    def forward(self, sent_id, mask):
        batch_size = len(sent_id)  # this must be independent from the `expected` batch_size
        x = self.bert(sent_id, attention_mask=mask)[1][-2][:,:,:]  # semi-last hidden layer, [CLS] token
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x.reshape(batch_size, 2)

    def evaluate(self, test_dataloader, device=torch.device("cpu")):
        """
        Evaluate the model on a dataloader defined as:
          test_data = TensorDataset(test_seq, test_mask, test_y)
          test_sampler = RandomSampler(test_data)
          test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
        """
        print("\nEvaluating...")
        self.eval()
        total_loss = 0
        total_preds, total_labels = np.array([]), np.array([])
        for step,batch in enumerate(test_dataloader):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))
            batch = [t.to(device) for t in batch]
            sent_id, mask, labels = batch
            with torch.no_grad():
                preds = self(sent_id, mask)
                loss = nn.NLLLoss()(preds,labels)
                total_loss = total_loss + loss.item()
                preds = preds.detach().cpu().numpy()
                total_preds = np.append(total_preds, [np.argmax(p) for p in preds], axis=0)
                total_labels = np.append(total_labels, labels.cpu().detach().numpy(), axis=0)
        avg_loss = total_loss / len(test_dataloader) 
        accuracy = np.mean([1 if p==l else 0 for p,l in zip(total_preds, total_labels)])
        return avg_loss, accuracy

class SentimentRoberta(nn.Module):
    """
    This is thought as a sentiment analysis classifier `head` for
     the transformers.BertForMaskedLM module. An example of usage is 
     provided in finetune_bert_sst.py.
    """
    def __init__(self, bert, maxlen):      
        super(SentimentRoberta, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(1024*maxlen, 256)
        self.fc2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten(1, -1)

    # Need to return only the softmax output
    def forward(self, sent_id, mask):
        batch_size = len(sent_id)  # this must be independent from the `expected` batch_size
        x = self.bert(sent_id, attention_mask=mask)[0][0:]
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x.reshape(batch_size, 2)

    def evaluate(self, test_dataloader, device=torch.device("cpu")):
        """
        Evaluate the model on a dataloader defined as:
          test_data = TensorDataset(test_seq, test_mask, test_y)
          test_sampler = RandomSampler(test_data)
          test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
        """
        print("\nEvaluating...")
        self.eval()
        total_loss = 0
        total_preds, total_labels = np.array([]), np.array([])
        for step,batch in enumerate(test_dataloader):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_dataloader)))
            batch = [t.to(device) for t in batch]
            sent_id, mask, labels = batch
            with torch.no_grad():
                preds = self(sent_id, mask)
                loss = nn.NLLLoss()(preds,labels)
                total_loss = total_loss + loss.item()
                preds = preds.detach().cpu().numpy()
                total_preds = np.append(total_preds, [np.argmax(p) for p in preds], axis=0)
                total_labels = np.append(total_labels, labels.cpu().detach().numpy(), axis=0)
        avg_loss = total_loss / len(test_dataloader) 
        accuracy = np.mean([1 if p==l else 0 for p,l in zip(total_preds, total_labels)])
        return avg_loss, accuracy

