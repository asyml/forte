# Pipeline Demo

The examples in this folder

- Build a pipeline using third-party tools like NLTK and StanfordNLP. 

- Demonstrate integration of processors developed using third-party tools into a single NLP 
pipeline.


# Description

`process_string_example.py` and `process_dataset_example.py` build the following pipeline

Reader -> NLTKSentenceSegmenter -> NLTKWordTokenizer -> NLTKPOSTagger -> NER Predictor -> 
SRL Predictor

The configuration for NER and SRL Predictor are read from `config.yml` file and the processors are 
initialized during the pipeline creation. To see the pipeline in action, run 

```bash
python pipeline_string_example.py
```

In `process_dataset_example.py`, we show the use of `process_dataset()` method of our pipeline 
which is used to read text files from a directory as data packs. To run this example,

```bash
python pipeline_dataset_example.py --data-dir data
```

In the above example, we read the files from `data/` folder.

`stanford_nlp_pipeline.py` example builds the following pipeline using Stanford NLP library

Reader -> Tokenizer -> POS Tagger -> Lemmatizer -> Dependency Parser

We run this pipeline on an English as well as French sentence. To see this action, run

```bash
python stanford_nlp_pipeline.py
``` 

# Quick Start

## Install the dependencies

- To run NLTK processors, please install NLTK library using

```bash
pip install nltk
```

- NER and SRL processors are built using texar-pytorch. Please follow the guidelines here 
https://github.com/asyml/texar-pytorch#installation to install it.

- To run `stanford_nlp_pipeline.py`, install Stanford NLP library using

```bash
pip install stanfordnlp
```




