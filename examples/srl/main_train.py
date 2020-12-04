# Copyright 2020 The Forte Authors. All Rights Reserved.
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
from typing import Iterator, Dict, List

from texar.torch.data import Batch
from torch import Tensor
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from forte.data.converter.feature import Feature
from forte.data.extractor.extractor import TextExtractor, BaseExtractor, \
    CharExtractor
from forte.data.types import DATA_INPUT, DATA_OUTPUT
from ft.onto.base_ontology import Sentence, Token, PredicateLink
from forte.train_preprocessor import TrainPreprocessor
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.models.srl_new.model import LabeledSpanGraphNetwork

logger = logging.getLogger(__name__)


def create_model(schemes: Dict[str, Dict[str, BaseExtractor]]) -> \
        LabeledSpanGraphNetwork:
    text_extractor: BaseExtractor = schemes["text_tag"]["extractor"]
    char_extractor: BaseExtractor = schemes["char_tag"]["extractor"]
    model = LabeledSpanGraphNetwork(word_vocab=text_extractor.get_dict(),
                                    char_vocab_size=char_extractor.size())

    return model


def train(model: LabeledSpanGraphNetwork,
          optim: Optimizer,
          batch: Batch) -> \
        LabeledSpanGraphNetwork.ReturnType:
    char_tensor: Tensor = batch["char_tag"]["tensor"]
    char_masks: List[Tensor] = batch["char_tag"]["mask"]
    text_tensor: Tensor = batch["text_tag"]["tensor"]
    text_mask: Tensor = batch["text_tag"]["mask"][0]
    raw_text_features: Feature = batch["raw_text_tag"]["features"]
    pred_link_features: List[Feature] = batch["pred_link_tag"]["features"]

    text: List[List[str]] = raw_text_features.unroll()[0]

    optim.zero_grad()

    output: LabeledSpanGraphNetwork.ReturnType = \
        model(text=text,
              char_batch=char_tensor,
              char_masks=char_masks,
              text_batch=text_tensor,
              text_mask=text_mask,
              srl_features=pred_link_features)

    output.loss.backward()
    optim.step()

    return output


num_epochs = 10
lr = 0.01
momentum = 0.9
nesterov = True

tp_request = {
    "scope": Sentence,
    "schemes": {
        "text_tag": {
            "entry_type": Token,
            "repr": "text_repr",
            "vocab_method": "indexing",
            "type": DATA_INPUT,
            "extractor": TextExtractor,
            "need_pad": True
        },
        "char_tag": {
            "entry_type": Token,
            "repr": "char_repr",
            "vocab_method": "indexing",
            "type": DATA_INPUT,
            "extractor": CharExtractor,
            "need_pad": True
        },
        "raw_text_tag": {
            "entry_type": Token,
            "repr": "text_repr",
            "vocab_method": "raw",
            "type": DATA_INPUT,
            "extractor": TextExtractor,
            "need_pad": False
        },
        "pred_link_tag": {  # predicate link
            "entry_type": PredicateLink,
            "attribute": "ner_type",
            "based_on": Token,
            "strategy": "BIO",
            "vocab_method": "indexing",
            "type": DATA_OUTPUT,
            "extractor": ...,
            "need_pad": False
        }
    }
}

tp_config = {
    "preprocess": {
        "pack_dir": "data/train/"
    },
    "dataset": {
        "batch_size": 10
    }
}

srl_train_reader = OntonotesReader()

train_preprocessor = TrainPreprocessor(train_reader=srl_train_reader,
                                       request=tp_request,
                                       config=tp_config)

model: LabeledSpanGraphNetwork = \
    create_model(schemes=train_preprocessor.feature_resource["schemes"])

optim: Optimizer = SGD(model.parameters(),
                       lr=lr,
                       momentum=momentum,
                       nesterov=nesterov)

logger.info("Start training.")
epoch = 0
while epoch < num_epochs:
    epoch += 1

    # Get iterator of preprocessed batch of train data
    train_batch_iter: Iterator[Batch] = \
        train_preprocessor.get_train_batch_iterator()

    for batch in tqdm(train_batch_iter):
        train_output: LabeledSpanGraphNetwork.ReturnType = \
            train(model, optim, batch)

        logger.info(f"{epoch}th Epoch training, "
                    f"loss: {train_output.loss}")
