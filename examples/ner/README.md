# NOTE: DEPRECATED
this example will be deprecated and we are moving them to the new training 
interface at `examples/tagging`.

# Introduction

This is an implementation of CNN-BiLSTM-CRF model, built on top of Texar and Pytorch. It's fully 
compatible with python 3.6 and Pytorch 1.0.1.


There are several minor modifications over the official codebase released by the author:

1. The official code [peaks the validation set and test set to build the 
word vocabulary](https://github.com/XuezheMax/NeuroNLP2/blob/2b9a0ea6ec9e1021660b29cdcd74c66824dd0e8c/neuronlp2/io/conll03_data.py#L33),
and initialize these word embeddings with Glove. We think this is not a 
general practice. So we merge the vocabulary of training set and Glove's 
directly, which will not affect the performance.

2. The official code [puts sentences with different lengths into buckets with staircase interval](https://github.com/XuezheMax/NeuroNLP2/blob/master/neuronlp2/io/conll03_data.py#L178), 
sample batches based on a frequency probability, and use a presumed batch size. For dynamic 
batching, we define the `batch_size_tokens` as `the number of word tokens in a batch` and uses 
`torchtext.data.iterator.pool` function.

3. We randomly pick a random seed for this codebase to make the result reproducible.

4. For the conditional random field model, the model with best performance provided by the author 
adopts a projection matrix (with shape `[hidden_state, label_cnt, label_cnt]`) to project the 
hidden state of each token to the label compatibility space, which is called `bigram` in the 
[code repository](https://github.com/XuezheMax/NeuroNLP2/blob/2b9a0ea6ec9e1021660b29cdcd74c66824dd0e8c/neuronlp2/nn/modules/crf.py#L34).
While we choose to define the label compatibility score as a matrix (with shape `[label_cnt, 
label_cnt]`), which is a more general definition as far as we know. Also, this definition brings 
less parameters. Fortunately, this implementation is provided by AllenNLP already so we don't need 
to rebuild the wheel. We don't apply any label transition restriction since we believe the model 
should be able to learn such restriction from data itself.

# Quick Start

## Getting the data

- We use the CoNLL-2003 NER dataset for this example. Please refer to [https://www.clips.uantwerpen.be/conll2003/ner/](https://www.clips.uantwerpen.be/conll2003/ner/)
for instructions on how to download the dataset. The `config_data.yml` file looks for the data in
`ner_data/conll03_english/{train/dev/test}`. If the structure of your dataset 
is different, please update the YAML file accordingly.

- We use GloVE vectors for word embeddings. Specifically, we use GloVE vectors trained on 6 
billion tokens. Download them here - [GloVE](http://nlp.stanford.edu/data/glove.6B.zip). 
Download and extract the folder in `ner_data/`.


 
## Install the dependencies

You need to install the [Texar-pytorch](https://github.com/asyml/texar-pytorch) first.

For data part, put the CONLL03 english data in the corresponding directory (`train_path`, 
`val_path`, and `test_path`) specified in `config_data.yml`. The data format is the same as the 
official codebase (https://github.com/XuezheMax/NeuroNLP2#data-format).
 
Then simply run

 ```bash
python main_train.py
```

The resource will be saved under `resource_dir` (specified in `config_model.yml`) and the model will
be saved in `model_path` (also specified in `config_model.yml`).

To run prediction using a trained model, run

```bash
python main_predict.py
```

In the above script, we first load the resource then create a pipeline with CoNLL03 Reader and 
Predictor. We then run this pipeline on the test dataset.

## Downloading the trained models

If you want to just try out the prediction example, we provide a script to download all the 
trained models for you. Just run

```bash
python download_models.py
```

This script will download all the necessary models in `resources/` folder. After this, you can 
run the prediction as before i.e.,

```bash
python main_predict.py
```

# Results

The default configuration provided in `config_data.yml` runs for 200 epochs and takes around 5.5 
hours to train on a single GeForce GTX 1080 Ti with 11GB of GPU memory. You should be able to 
achieve the following results during training. 

```bash
INFO:forte.trainer.ner_trainer:Best val acc:  98.900, precision: 95.040, recall: 94.720, F1: 94.880, epoch=129
INFO:forte.trainer.ner_trainer:Best test acc:  98.010, precision:  91.430, recall:  91.450, F1:  91.440, epoch=129
```    