# Semantic robustness for NLP models.

This repository allows to train and test models against linguistic phenomena.
So far we provide benchmarks for sentiment analysis and the following phenomena:
- `negation`: a positive sentence is negate or vice versa.
- `mixed sentiment`: a sentence that contains positive and negative terms but only one of the two is related to the classification.
- `sarcasm`.

## Train a model:
Please go to the \train directory and run `train_{BERT, sst}.py` via python3 command (e.g., python3 train_sst.py).
Neural networks trained in this way are stored inside the \models directory, each in the respective folder depending on the architecture (FC, CNN, attention, lstm).
The training parameters are declared at the beginning of each file (`# Training parameters` comment), while it is possible to augment the training dataset with linguistical samples (either from our template based generator or from [1]) by changing the parameters `augment_rule1` and `augment_rule2` just after the previous code.
### An argparse version of this script will be added soon. 

## Test a model:
Please go to \verify folder and run `python3 semantic_robustness_{bert, nn}.py`.
The verificatin parameters are declared at the beginning of each file (`# Training parameters` comment).

If you want to cite the code or the paper, please use the following bibtex (soon to come the AAAI bibtex):
```
@inproceedings{la2022king,
  title={The king is naked: on the notion of robustness for natural language processing},
  author={La Malfa, Emanuele and Kwiatkowska, Marta},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={10},
  pages={11047--11057},
  year={2022}
}
```

