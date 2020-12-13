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

import torch
from texar.torch.data import Batch
from torch import Tensor
from torch.optim import SGD
from torch.optim.optimizer import Optimizer

from forte.data.converter.feature import Feature
from forte.data.extractor.attribute_extractor import AttributeExtractor
from forte.data.extractor.base_extractor import BaseExtractor
from forte.data.extractor.char_extractor import CharExtractor
from forte.data.extractor.link_extractor import LinkExtractor
from ft.onto.base_ontology import Sentence, Token, PredicateLink
from forte.train_preprocessor import TrainPreprocessor
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.models.srl_new.model import LabeledSpanGraphNetwork
from forte.models.srl_new import data

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def create_model(schemes: Dict[str, Dict[str, BaseExtractor]]) -> \
        LabeledSpanGraphNetwork:
    text_extractor: BaseExtractor = schemes["text_tag"]["extractor"]
    char_extractor: BaseExtractor = schemes["char_tag"]["extractor"]
    link_extractor: BaseExtractor = schemes["pred_link_tag"]["extractor"]

    # This is used when enforce_constraint is enabled during decoding
    core_args_ids: Dict[str, int] = {}
    for core_arg in LabeledSpanGraphNetwork._CORE_ARGS:
        core_args_ids[core_arg] = link_extractor.element2repr(core_arg)

    model = LabeledSpanGraphNetwork(word_vocab=text_extractor.get_dict(),
                                    char_vocab_size=char_extractor.size(),
                                    label_vocab_size=link_extractor.size(),
                                    core_args_ids=core_args_ids)

    return model


def train(model: LabeledSpanGraphNetwork,
          optim: Optimizer,
          batch: Batch) -> \
        LabeledSpanGraphNetwork.ReturnType:
    char_tensor: Tensor = batch["char_tag"]["tensor"]
    char_masks: List[Tensor] = batch["char_tag"]["masks"]
    text_tensor: Tensor = batch["text_tag"]["tensor"]
    text_mask: Tensor = batch["text_tag"]["masks"][0]
    raw_text_features: List[Feature] = batch["raw_text_tag"]["features"]
    pred_link_features: List[Feature] = batch["pred_link_tag"]["features"]

    text: List[List[str]] = []
    for feature in raw_text_features:
        text.append(feature.data[0])

    optim.zero_grad()

    output: LabeledSpanGraphNetwork.ReturnType = \
        model(text=text,
              char_batch=char_tensor,
              char_masks=char_masks,
              text_batch=text_tensor,
              text_mask=text_mask,
              srl_features=pred_link_features)

    output["loss"].backward()
    optim.step()

    return output


def predict_forward_fn(model: LabeledSpanGraphNetwork, batch: Dict) -> Dict:
    char_tensor: Tensor = batch["char_tag"]["tensor"]
    char_masks: List[Tensor] = batch["char_tag"]["masks"]
    text_tensor: Tensor = batch["text_tag"]["tensor"]
    text_mask: Tensor = batch["text_tag"]["masks"][0]
    raw_text_features: List[Feature] = batch["raw_text_tag"]["features"]

    text: List[List[str]] = []
    for feature in raw_text_features:
        text.append(feature.data[0])

    # TODO: test enable enforce_constriant
    model_output: List[Dict[int, List[data.Span]]] = \
        model.decode(text=text,
                     char_batch=char_tensor,
                     char_masks=char_masks,
                     text_batch=text_tensor,
                     text_mask=text_mask)

    output: List[Dict] = []
    for model_output_i in model_output:
        # TODO: use extractor specified name
        output_i = {
            "data": [],
            "parent_unit_span": [],
            "child_unit_span": []
        }
        for predicate_id, span in model_output_i.items():
            span: data.Span
            output_i["data"].append(span.label)
            output_i["parent_unit_span"].append((predicate_id, predicate_id+1))
            output_i['child_unit_span'].append((span.start, span.end))

    return {'pred_link_tag': output}


num_epochs = 5
lr = 0.01
momentum = 0.9
nesterov = True
batch_size = 64

train_path = "data/train_tiny/"

tp_request = {
    "scope": Sentence,
    "schemes": {
        "text_tag": {
            "entry_type": Token,
            "attribute_get": "text",
            "vocab_method": "indexing",
            "type": TrainPreprocessor.DATA_INPUT,
            "extractor": AttributeExtractor,
            "need_pad": True
        },
        "char_tag": {
            "entry_type": Token,
            "vocab_method": "indexing",
            "type": TrainPreprocessor.DATA_INPUT,
            "extractor": CharExtractor,
            "need_pad": True
        },
        "raw_text_tag": {
            "entry_type": Token,
            "attribute_get": "text",
            "vocab_method": "raw",
            "type": TrainPreprocessor.DATA_INPUT,
            "extractor": AttributeExtractor,
            "need_pad": False
        },
        "pred_link_tag": {  # predicate link
            "entry_type": PredicateLink,
            "attribute": "arg_type",
            "based_on": Token,
            "strategy": "BIO",
            "vocab_method": "indexing",
            "type": TrainPreprocessor.DATA_OUTPUT,
            "extractor": LinkExtractor,
            "need_pad": False
        }
    }
}

tp_config = {
    "preprocess": {
        "pack_dir": train_path
    },
    "dataset": {
        "batch_size": batch_size
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
train_loss: float = 0.0
train_total: int = 0

while epoch < num_epochs:
    epoch += 1

    # Get iterator of preprocessed batch of train data
    train_batch_iter: Iterator[Batch] = \
        train_preprocessor.get_train_batch_iterator()

    for batch in train_batch_iter:
        train_output: LabeledSpanGraphNetwork.ReturnType = \
            train(model, optim, batch)
        train_loss += train_output["loss"]
        train_total += 1

    logger.info(f"{epoch}th Epoch training, "
                f"loss: {train_loss / train_total}")

    train_loss = 0.0
    train_total = 0

# Save training state to disk
# train_preprocessor.save_state("train_state.pkl")
# torch.save(model, "model.pt")
