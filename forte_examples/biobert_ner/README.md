# BioBERT NER Predictor Demo

The examples in this folder

- Defines an pipeline using NLTK and BioBERT to extract medical name entities

- Provides an NER prediction model based on BioBERT v1.1 and fine-tuned on the NCBI-disease dataset


# Description

## Install the dependencies

- To use NLTK processors, please install NLTK library using

```bash
pip install nltk
```

- The NER predictor is built using biobert-pytorch and transformers. Please follow the guidelines here 
https://github.com/dmis-lab/biobert-pytorch to install it.


## Downloading the models

In this pipeline, we use an NER model fine-tuned on the NCBI-disease dataset, which predicts 
named entities corresponding to diseases, symptoms and related concepts. 
Before running the pipeline, we need to download the models

```bash
python download_models.py 
```

**Note**: The above script will save the model in `resources/NCBI-disease`. Use `--path` option to save the 
model into a different directory.

## Running the example

`ner_predict_example.py` builds the following pipeline for NER prediction

Reader -> NLTK SentenceSegmenter -> BERT Tokenizer -> BioBERT NER Predictor

The configuration for BERT Tokenizer and BioBERT NER Predictor are read from `config.yml` file and the processors are 
initialized during the pipeline creation. To see the pipeline in action, run 

```bash
python ner_predict_example.py
```

