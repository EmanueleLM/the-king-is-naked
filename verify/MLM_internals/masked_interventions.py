# pip install -U pytorch-pretrained-bert
import copy as cp
import itertools
import numpy as np
import random
import torch

from transformers import BertTokenizer

from mlm_text_utils import random_combination

import logging
logging.basicConfig(level=logging.INFO)

#DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'

def mapIndexValues(a, b):
    """
    Extract elements in a (list) using indices in b (list:int)
    """
    out = map(a.__getitem__, b)
    return list(out)

def tokenize_text(text, return_indices=False, use_bert=True, TOKENIZER=None):
    """
    TODO: complete documentation
    use_bert:boolean
     (optional) use BERT tokenizer to split the input string, otherwise use split function.
    TOKENIZER:bert-tokenizer
      (optional) In your code, the tokenizer should be set as a valid tokenizer, e.g.:
      from transformers import BertTokenizer;TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        or
     from transformers import AutoTokenizer;TOKENIZER = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')"
    """
    if use_bert is True:
        tokenized_text = TOKENIZER.tokenize(text)
    else:
        tokenized_text = text.split(' ')
    if return_indices is True:
        return TOKENIZER.convert_tokens_to_ids(tokenized_text)
    return tokenized_text

def softmax(p):
    """
    p is a bidimensional vector and will be normalized `column-wise`
    """
    exp_p = np.exp(p)
    exp_p = exp_p/np.sum(exp_p, axis=1, keepdims=True)
    return exp_p

def targeted_interventions(model, masked_text, original_text, topk=5, mask='[MASK]', predicts_only=[], words_per_concept=10, budget=100, device='cpu', TOKENIZER=None):
    """
    Predicts a subset of masked tokens in a text, generating subsitutions from the first token on left to the last on the right.
    model:function 
     returns a prediction on an input text
    masked_text:list
     tokenized text (tokens in a concept are masked)
    original_text:list
     tokenized text (tokens in a concept are not masked): this is used when not all the words in a concepts are masked
    masked_text:list
     tokenized text (with mask tokens)
    topk:int
     (optional) number of replacements returned per-word
    predicts_only:list
     (optional) list of lists of indices from the masked_text on which BERT predicts the substitutions 
      Substitutions in lists inside the main list are considered independent from each other.
      All the other `masks` are ignored. If predicts_only==[], it predicts on every `masked` tokens.
    words_per_concept:int
     (optional) limit the number of words that are evaluated in each concept (i.e., each list)
      (5 is a good measure, greater values melt the machine).
    budget:int
     (optional) maximum number of permutations returned at each iteration
    TOKENIZER:bert-tokenizer
      (optional) In your code, the tokenizer should be set as a valid tokenizer, e.g.:
      from transformers import BertTokenizer;TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        or
     from transformers import AutoTokenizer;TOKENIZER = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')"
    """
    global DEVICE
    flatten_indices_predicts_only, flatten_indices_nomask = [], []
    flatten_indices_predicts_only += predicts_only[:words_per_concept]
    flatten_indices_nomask += predicts_only
    masked_indices = [i for i,w in enumerate(masked_text) if w == mask and (i in flatten_indices_predicts_only if predicts_only!=[] else True)]
    # If there is nothing to predict, return
    if len(masked_indices) == 0:
        return {}
    # Restore the original text when not predicting a masked token
    for i,_ in enumerate(original_text):
        if i not in flatten_indices_predicts_only and i in flatten_indices_nomask:
            masked_text[i] = original_text[i]
    # Re-generate masked indices but now in compact lists per-concept
    masked_indices = predicts_only[:words_per_concept]  # alternative name: `indices_predicts_only`
    # Tokenize text
    masked_texts = [[TOKENIZER.convert_tokens_to_ids(masked_text)] for _ in masked_indices]
    # Start genereating perturbations
    predicted_tokens = {k:[] for k in flatten_indices_predicts_only}
    #print(masked_indices)
    #print(masked_texts)
    for _,midx in enumerate(masked_indices):
        tmp = []
        #print(f"Processing index {midx} out of {single_concept_interventions}")
        #print(f"{len(masked_texts[scp_index][-1][:budget])} will be processed")
        for mt in masked_texts[-1][:budget]:
            #print(mt)
            segments_ids = [0] * len(mt)
            tokens_tensor = torch.tensor([mt]).to(DEVICE)
            segments_tensors = torch.tensor([segments_ids]).to(DEVICE)
            with torch.no_grad():
                predictions = model(tokens_tensor, segments_tensors)[0]
            #print(predictions.shape)
            #print(torch.topk(predictions[0, midx], topk).indices)
            for r in torch.topk(predictions[0, midx], topk).indices:
                mt_copy = cp.copy(mt)
                mt_copy[midx] = int(r)
                tmp += [mt_copy]
                #print(mt_copy)
                if len(tmp) > budget:
                    break
            if len(tmp) > budget:
                break
        masked_texts += [tmp]
    for mt in masked_texts[-1][:budget]:
        MIV = TOKENIZER.convert_ids_to_tokens(mapIndexValues(mt, masked_indices))
        for k, miv in zip(masked_indices, MIV):
            predicted_tokens[k] += [miv]
    #print(predicted_tokens)
    return predicted_tokens

def sequential_interventions(model, text, interventions, topk=1, combine_subs=False, mask="[MASK]", budget=250, device='cpu', TOKENIZER=None):
    """
    Use this function to perturb tokens in a text following a the order specified in an intervention list.
    Input:
    model:function
        function used to predict the output class of the input text
    text:list
        a list of words, already tokenized (but not masked)
    interventions:list
        list of lists (of indices) of words that are affected by the interventions. 
        The order of the lists specifies the relationship between parents and children. 
        In the same list each sub-item is a concept list (each are independent)
    topk:int
        (optional) number of replacements generated by BERT for each simulation
    combine_subs:boolean
        (optional) combine topk substitutions obtained for each word through their cartesian product
    mask:str
        (optional) token used to mask words to-be-predicted
    budget:int
     (optional) maximum number of permutations returned at each iteration
    TOKENIZER:bert-tokenizer
      (optional) In your code, the tokenizer should be set as a valid tokenizer, e.g.:
      from transformers import BertTokenizer;TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        or
     from transformers import AutoTokenizer;TOKENIZER = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')"
    Output:
    A list of substitutions, which in turn are lists of word-level tokens
    TODO: any parent influences all the subs, this code is possibly wrong (e.g., in case of disjoint children)
    """
    subs = [cp.copy(text)]
    subs[0] = [(s if i not in interventions else mask) for i,s in enumerate(subs[0])]
    #print(subs)
    new_subs = []
    for sub in subs:
        tmp_text = cp.copy(sub)
        dict_subs = targeted_interventions(model, tmp_text, text, topk=topk, mask=mask, predicts_only=interventions, budget=budget, device=device, TOKENIZER=TOKENIZER)
        #print(dict_subs)
        dict_subs_keys, dict_subs_values = list(dict_subs.keys()), list(dict_subs.values())
        print(dict_subs_keys)
        print(dict_subs_values)
        combinations = (zip(*dict_subs_values) if combine_subs is False else itertools.islice(itertools.product(*dict_subs_values), budget))  # set limit to cartesian product
        for el in combinations:
            tmp_sub = cp.copy(tmp_text)
            for ii, idx in enumerate(dict_subs_keys):
                tmp_sub[idx] = el[ii]  
            new_subs += [tmp_sub]
            if len(new_subs) > budget:
                break
        if len(new_subs) > budget:
            break
    subs = cp.copy(new_subs)
    new_subs = []
    return subs[:budget]

def most_likely_token_bottom_up(model, masked_text, original_text, mask='[MASK]', device='cpu', TOKENIZER=None):
    """
    Predicts the most likely token from a text, comparing it to the occluded version
     returns the index i=argmax_i(P(m_i \in masked_text | masked_text, original_text))
    masked_text:list
     tokenized text (tokens in a concept are masked)
    original_text:list
     tokenized text (tokens in a concept are not masked): this is used when not all the words in a concepts are masked
    TOKENIZER:bert-tokenizer
      (optional) In your code, the tokenizer should be set as a valid tokenizer, e.g.:
      from transformers import BertTokenizer;TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
        or
     from transformers import AutoTokenizer;TOKENIZER = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')"
    """
    global DEVICE
    print(f"Original sequence: `{original_text}`")
    print(f"Masked sequence: `{masked_text}`")
    # Tokenize text and masked text
    indexed_original_text = [TOKENIZER.convert_tokens_to_ids(original_text)]
    indexed_masked_text = [TOKENIZER.convert_tokens_to_ids(masked_text)]
    predict = []
    if len(indexed_masked_text) != len(indexed_original_text):
        print(f"[logger-WARNING] variables indexed_masked_texts and indexed_original_text are of different length, {len(indexed_masked_text)}!={len(indexed_original_text)}")
    for o,m,i in zip(original_text, masked_text, range(len(original_text))):
        #print(o,m,i)
        if m != mask or o == '[CLS]' or o == '[SEP]':
            pass
        else:
            predict += [i]
    #print(f"Can predict on {predict}")
    # Find the token that maximises the likelihood of being generated by the MLM
    segment_ids = [0] * len(indexed_original_text)
    segments_tensors = torch.tensor([segment_ids]).to(DEVICE)
    tokens_masked_tensor = torch.tensor(indexed_masked_text).to(DEVICE)
    tokens_original_tensor = torch.tensor([indexed_original_text]).to(DEVICE)
    #print("#", indexed_masked_text)
    with torch.no_grad():
        predictions_masked = model(tokens_masked_tensor, segments_tensors)[0]  # predict
        predictions_masked_numpy = predictions_masked.numpy()
    if predictions_masked_numpy.ndim == 3:
        predictions_masked_numpy = predictions_masked_numpy[0]
    prediction_odds = softmax(predictions_masked_numpy)
    #print(prediction_odds.shape)
    #print(prediction_odds.sum(axis=1))
    # predict the most likely token
    max_odd, most_likely_idx = 0., None
    for i,idx in enumerate(indexed_original_text[0]):
        #print(idx)
        if i in predict:
            print(f"Prob of word {original_text[i]}, position {i}, token {idx}, is {prediction_odds[i,idx]}")
            if max_odd <= prediction_odds[i,idx]:
                max_odd = prediction_odds[i,idx]
                most_likely_idx = i
    return most_likely_idx

