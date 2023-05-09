#  Copyright 2020 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import numpy as np
import torch
import yaml

from torch import nn
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from texar.torch.data import Batch
from tqdm import tqdm
from typing import Iterator, Dict, Any
from forte.common.configuration import Config
from forte.data.extractor.attribute_extractor \
    import AttributeExtractor
from forte.data.extractor.base_extractor \
    import BaseExtractor
from forte.train_preprocessor import TrainPreprocessor
from forte.data.readers.imdb_reader import IMDBReader
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence, Token, Entry
from texar.torch.modules.embedders import WordEmbedder
from examples.classification_example.cnn import CNN_Classifier
from forte.processors.base.data_augment_processor import ReplacementDataAugmentProcessor
from texar.torch.modules.classifiers.bert_classifier import BERTClassifier
from examples.classification_example.util import pad_each_bach

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")


def construct_word_embedding_table(embed_dict, extractor: BaseExtractor):
    embedding_dim = list(embed_dict.values())[0].shape[-1]

    scale = np.sqrt(3.0 / embedding_dim)
    table = np.empty(
        [extractor.size(), embedding_dim], dtype=np.float32
    )
    oov = 0
    for word, index in extractor.items():
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


def create_model(text_extractor: AttributeExtractor,
                 config: Config, in_channels: int):
    embedding_dict = {}
    for word, index in text_extractor.items():
        embedding_dict[word] = torch.tensor([0.0 for i in range(100)])

    word_embedding_table = \
        construct_word_embedding_table(embedding_dict, text_extractor)

    model: nn.Module = \
        CNN_Classifier(in_channels=in_channels,
                       word_embedding_table=word_embedding_table)

    if config.config_model.model == "bert":
        model: nn.Module = BERTClassifier()

    return model, word_embedding_table


def train(model: nn.Module, optim: Optimizer, batch: Batch, max_sen_length: int):
    word = batch["text_tag"]["data"]
    labels = batch["label_tag"]["data"]
    optim.zero_grad()

    logits, pred = None, None

    if config.config_model.model == "cnn":
        logits, pred = model(batch)

    if config.config_model.model == "bert":
        mask = batch["text_tag"]["masks"][0]
        logits, pred = model(pad_each_bach(word, max_sen_length),
                             torch.sum(mask, dim=1))

    labels_1D = torch.squeeze(labels)
    true_one_batch = (labels_1D == pred).sum().item()
    loss = criterion(logits, labels_1D)

    loss.backward()
    optim.step()

    batch_train_err = loss.item() * batch.batch_size

    return batch_train_err, true_one_batch


# All the configs
config_data = yaml.safe_load(open("config_data.yml", "r"))
config_model = yaml.safe_load(open("config_model.yml", "r"))

config = Config({}, default_hparams=None)
config.add_hparam('config_data', config_data)
config.add_hparam('config_model', config_model)

# Generate request
text_extractor: AttributeExtractor = \
    AttributeExtractor(config={"entry_type": Token,
                               "vocab_method": "indexing",
                               "attribute": "text"})


class SentimentExtractor(AttributeExtractor):
    def get_attribute(self, entry: Entry, attr: str):
        if entry.sentiment["positive"] == 1.0:
            return "positive"
        else:
            return "negative"

    def set_attribute(self, entry: Entry, attr: str, value: Any):
        if value == "positive":
            entry.sentiment = {
                "positive": 1.0,
                "negative": 0.0,
            }
        else:
            entry.sentiment = {
                "positive": 0.0,
                "negative": 1.0,
            }


label_extractor: AttributeExtractor = \
    SentimentExtractor(config={"entry_type": Sentence,
                               "vocab_method": "indexing",
                               "need_pad": False,
                               "vocab_use_unk": False,
                               "attribute": "sentiment"})
tp_request: Dict = {
    "scope": Sentence,
    "schemes": {
        "text_tag": {
            "type": TrainPreprocessor.DATA_INPUT,
            "extractor": text_extractor
        },
        "label_tag": {
            "type": TrainPreprocessor.DATA_OUTPUT,
            "extractor": label_extractor
        }
    }
}


# Default settings can be found here:
# https://texar-pytorch.readthedocs.io/en/latest/code/data.html#texar.torch.data.DatasetBase.default_hparams
tp_config = {
    "preprocess": {
        "device": device.type
    },
    "dataset": {
        "batch_size": config.config_data.batch_size_tokens
    }
}

processor_config = {
    'augment_entry': "ft.onto.base_ontology.Token",
    'other_entry_policy': {
        "kwargs": {
            "ft.onto.base_ontology.Sentence": "auto_align"
        }
    },
    'type': 'data_augmentation_op',
    'data_aug_op': 'tests.forte.processors.base.data_augment_replacement_processor_test.TmpReplacer',
    "data_aug_op_config": {
        'kwargs': {}
    },
    'augment_pack_names': {
        'kwargs': {}
    }
}

imdb_train_reader = IMDBReader()

pl = Pipeline()
pl.set_reader(imdb_train_reader)
if config.config_data.data_aug:
    pl.add(ReplacementDataAugmentProcessor(), processor_config)
pl.initialize()

datapack_generator = pl.process_dataset(config.config_data.train_path)


train_preprocessor = TrainPreprocessor(pack_generator=datapack_generator,
                                       request=tp_request,
                                       config=tp_config)


max_sen_length = 0
train_batch_iter: Iterator[Batch] = \
        train_preprocessor.get_train_batch_iterator()

# Fing the max sentence length in the whole dataset
for batch in tqdm(train_batch_iter):
    max_sen_length = max(batch["text_tag"]["data"].size()[1],
                         max_sen_length)

model, word_embedding_table = \
    create_model(text_extractor=text_extractor,
                 config=config, in_channels=max_sen_length)

word_embedder = WordEmbedder(init_value=word_embedding_table)

model.to(device)

criterion = nn.CrossEntropyLoss()

optim: Optimizer = SGD(model.parameters(),
                       lr=config.config_model.learning_rate,
                       momentum=config.config_model.momentum,
                       nesterov=True)

epoch = 0
train_err: float = 0.0
train_total: float = 0.0
train_sentence_len_sum: int = 0

logger.info("Start training.")

while epoch < config.config_data.num_epochs:
    epoch += 1

    train_batch_iter: Iterator[Batch] = \
        train_preprocessor.get_train_batch_iterator()

    true_total = 0
    train_total_One_Epoch = 0

    for batch in tqdm(train_batch_iter):
        batch_train_err, true_one_batch = train(model, optim, batch, max_sen_length)

        train_err += batch_train_err
        train_total += batch.batch_size
        true_total += true_one_batch
        train_total_One_Epoch += batch.batch_size

    logger.info(f"{epoch}th Epoch training, "
                f"total number of examples: {train_total}, "
                f"Train Accuracy: {(true_total / train_total_One_Epoch):0.3f}, "
                f"loss: {(train_err / train_total):0.3f}")

# Save training result to disk
# train_preprocessor.save_state(config.config_data.train_state_path)
# torch.save(model, config.config_model.model_path)
