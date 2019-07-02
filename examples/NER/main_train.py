"""
In this pipeline, we need to
1.  Build ner_data packs from multiple ner_data sets
2.  create vocabulary over the training ner_data packs (With processor)
    - Should create the vocabulary over word level, character level, NER tags.
3.  create a model processor
4.  Use trainer processor to perform training
        trainer only accepts ner_data and
5. Use validation processor on validation dataset
6.

The training logic (where to save the model, update learning rate) is held
    in pipeline?
"""
import importlib
from nlp.pipeline.data.readers.conll03_reader import CoNLL03Reader
from nlp.pipeline.common.resources import Resources
from nlp.pipeline.trainer.trainers.ner_trainer import CoNLLNERTrainer
from nlp.pipeline.processors.impl.ner_predictor import (
    CoNLLNERPredictor,
    CoNLLNEREvaluator,
)
from nlp.pipeline.train_pipeline import TrainPipeline

from vocabulary_processor import CoNLL03VocabularyProcessor, Alphabet
from model_factory import BiRecurrentConvCRF
from utils import load_glove_embedding
import torch
from torch.optim import SGD

config_data = importlib.import_module("config_data")
config_model = importlib.import_module("config_model")

reader = CoNLL03Reader(lazy=False)

embedding_path = config_model.embedding_path

train_reader = reader.dataset_iterator(config_data.train_path)
val_reader = reader.dataset_iterator(config_data.val_path)
test_reader = reader.dataset_iterator(config_data.test_path)
embedding_dict, embedding_dim = load_glove_embedding(embedding_path)

# Keep the vocabulary processor as a simple counter
vocab_processor = CoNLL03VocabularyProcessor()

word_cnt, char_cnt, pos_cnt, chunk_cnt, ner_cnt = vocab_processor.process(
    train_reader
)

word_alphabet = Alphabet("word", word_cnt)
char_alphabet = Alphabet("character", char_cnt)
pos_alphabet = Alphabet("pos", pos_cnt)
chunk_alphabet = Alphabet("chunk", chunk_cnt)
ner_alphabet = Alphabet("ner", ner_cnt)

print(f'before expanding, word_alphabet size:{word_alphabet.size()}')
# To facilitate model sharing between trainer and predictor, we build the model
# as an instance of resource
model = BiRecurrentConvCRF(
    word_alphabet, char_alphabet, ner_alphabet, embedding_dict, embedding_dim
)
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)
model = model.to(device=device)
optim = SGD(
    model.parameters(),
    lr=config_model.learning_rate,
    momentum=config_model.momentum,
    nesterov=True,
)

resources = Resources(
    word_alphabet=word_alphabet,
    char_alphabet=char_alphabet,
    pos_alphabet=pos_alphabet,
    chunk_alphabet=chunk_alphabet,
    ner_alphabet=ner_alphabet,
    embedding_dict=embedding_dict,
    embedding_dim=embedding_dim,
    config_model=config_model,
    config_data=config_data,
    model=model,
    optim=optim,
)

ner_trainer = CoNLLNERTrainer(config=None)
ner_predictor = CoNLLNERPredictor()

ner_evaluator = CoNLLNEREvaluator(config=None)

mypipeline = TrainPipeline(
    train_reader=reader,
    trainer=ner_trainer,
    predictor=ner_predictor,
    evaluator=ner_evaluator,
    dev_reader=reader,
    resource=resources,
)

mypipeline.train()
