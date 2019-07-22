import numpy as np
import yaml
import torch
from torch.optim import SGD
from texar.hyperparams import HParams

from examples.NER.model_factory import BiRecurrentConvCRF
from nlp.pipeline.common.resources import Resources
from nlp.pipeline.data.readers.conll03_reader import CoNLL03Reader
from nlp.pipeline.models.NER.utils import load_glove_embedding
from nlp.pipeline.models.NER.utils import set_random_seed
from nlp.pipeline.processors.impl.ner_predictor import (
    CoNLLNEREvaluator, CoNLLNERPredictor)
from nlp.pipeline.processors.impl.vocabulary_processor import (
    Alphabet, CoNLL03VocabularyProcessor)
from nlp.pipeline.train_pipeline import TrainPipeline
from nlp.pipeline.trainer.impl.ner_trainer import CoNLLNERTrainer


def construct_word_embedding_table(embed_dict, alphabet):
    embedding_dim = list(embed_dict.values())[0].shape[-1]

    scale = np.sqrt(3.0 / embedding_dim)
    table = np.empty(
        [alphabet.size(), embedding_dim], dtype=np.float32
    )
    oov = 0
    for word, index in alphabet.items():
        if word in embed_dict:
            embedding = embed_dict[word]
        elif word.lower() in embed_dict:
            embedding = embed_dict[word.lower()]
        else:
            embedding = np.random.uniform(
                -scale, scale, [1, embedding_dim]
            ).astype(np.float32)
            oov += 1
        table[index, :] = embedding
    return torch.from_numpy(table)


def main():
    config_data = yaml.safe_load(open("config_data.yml", "r"))
    config_model = yaml.safe_load(open("config_model.yml", "r"))
    # This is the configuration for the whole task
    # so there is not default_hparams
    config_data = HParams(config_data, default_hparams=None)
    config_model = HParams(config_model, default_hparams=None)

    set_random_seed(config_model.random_seed)

    reader = CoNLL03Reader(lazy=False)

    embedding_path = config_model.embedding_path

    train_reader = reader.dataset_iterator(config_data.train_path)
    unused_val_reader = reader.dataset_iterator(config_data.val_path)
    unused_test_reader = reader.dataset_iterator(config_data.test_path)
    embedding_dict = load_glove_embedding(embedding_path)

    # Keep the vocabulary processor as a simple counter
    vocab_processor = CoNLL03VocabularyProcessor()

    (word_cnt, char_cnt, unused_pos_cnt,
     unused_chunk_cnt, ner_cnt) = vocab_processor.process(train_reader)

    word_alphabet = Alphabet("word", word_cnt)
    char_alphabet = Alphabet("character", char_cnt)
    ner_alphabet = Alphabet("ner", ner_cnt)

    for word in embedding_dict:
        if word not in word_alphabet.instance2index:
            word_alphabet.add(word)

    word_alphabet.save(config_data["alphabet_directory"])
    char_alphabet.save(config_data["alphabet_directory"])
    ner_alphabet.save(config_data["alphabet_directory"])

    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device(
            "cpu")
    )

    word_embedding_table = construct_word_embedding_table(embedding_dict,
                                                          word_alphabet)

    print(f'word embedding table size:{word_embedding_table.size()}')
    normalize_func = vocab_processor.normalize_func

    model = BiRecurrentConvCRF(
        word_embedding_table, char_alphabet.size(), ner_alphabet.size(),
        config_model
    )

    model = model.to(device=device)

    optim = SGD(
        model.parameters(),
        lr=config_model.learning_rate,
        momentum=config_model.momentum,
        nesterov=True,
    )

    # To facilitate model sharing between trainer and predictor,
    # we build the model as an instance of resource

    resources = Resources(
        word_alphabet=word_alphabet,
        char_alphabet=char_alphabet,
        ner_alphabet=ner_alphabet,
        normalize_func=normalize_func,
        config_model=config_model,
        config_data=config_data,
        model=model,
        optim=optim,
        device=device
    )

    ner_trainer = CoNLLNERTrainer()
    ner_predictor = CoNLLNERPredictor()
    ner_evaluator = CoNLLNEREvaluator()

    mypipeline = TrainPipeline(
        train_reader=reader,
        trainer=ner_trainer,
        predictor=ner_predictor,
        evaluator=ner_evaluator,
        dev_reader=reader,
        resource=resources,
    )

    # the training configuration are specified in config_model
    mypipeline.train()


if __name__ == '__main__':
    main()
