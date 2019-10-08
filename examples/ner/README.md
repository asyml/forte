# Introduction

This is an implementation of CNN-BiLSTM-CRF model, built on top of Texar 
and Pytorch. It's fully compatible with python 3.6 and Pytorch 1.0.1.


There are several minor modifications over the official codebase released by 
the author:
1. The official code [peaks the validation set and test set to build the 
word vocabulary](https://github.com/XuezheMax/NeuroNLP2/blob/2b9a0ea6ec9e1021660b29cdcd74c66824dd0e8c/neuronlp2/io/conll03_data.py#L33),
and initialize these word embeddings with Glove. I think this is not a 
general practice. So I merge the vocabulary of training set and Glove's 
directly, which will not affect the performance.

2. The official code [put sentences with different lengths into buckets with staircase interval](https://github.com/XuezheMax/NeuroNLP2/blob/master/neuronlp2/io/conll03_data.py#L178), 
sample batches based on a frequency probability, and use a presumed batch 
size. For dynamic batching, I define the `batch_size_tokens` as `the number of 
word tokens in a batch` and uses `torchtext.data.iterator.pool` function.

3. I randomly pick a random seed for this codebase to make the result 
reproducible.

4. For the conditional random field model, the model with best performance 
provided by the author adopts a projection matrix (with shape `[hidden_state,
 label_cnt, label_cnt]`) to project the 
hidden state of each token to the label compatibility space, which is called 
`bigram` in the [code repository](https://github.com/XuezheMax/NeuroNLP2/blob/2b9a0ea6ec9e1021660b29cdcd74c66824dd0e8c/neuronlp2/nn/modules/crf.py#L34).
While I choose to define the label compatibility score as a matrix (with 
shape `[label_cnt, label_cnt]`), which is a more general definition as far as
 I know. Also, this definition brings less parameters. Fortunately this 
 implementation is provided by AllenNLP already so I 
 don't need to rebuild the wheel. I don't apply any label transition restriction 
 since I believe the model should be able to learn such restriction from data
 itself.

# Quick Start

## Install the dependencies

You need to install the [Texar-pytorch](https://github.com/asyml/texar-pytorch) first.


For data part, put the CONLL03 english data in the corresponding directory
 (`train_path`, `val_path`, and `test_path`)
 specified in `config_data.py`. The required data format is the same as the 
 official codebase(https://github.com/XuezheMax/NeuroNLP2#data-format).
 
Then simply run
 ```bash
python main_train.py
```

The Alphabets (for word, character, POS, Chunk and NER tags) will be saved in
 `alphabet_directory` specified in `config_data.py`  and the model will be 
 saved in `model_path` specified in `config_model.py`. The model output on 
 validation and test set will be saved in `tmp/val_output.txt` and 
 `tmp/test_output.txt`.