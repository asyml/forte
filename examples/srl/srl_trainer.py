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
from typing import Iterator, Dict, List
from torch import Tensor
from torch.optim import SGD
from torch.optim.optimizer import Optimizer
from texar.torch.data import Batch

from forte.models.srl_new.model import LabeledSpanGraphNetwork
from forte.data.converter import Converter, Feature
from forte.predictor import Predictor
from forte.data.readers import OntonotesReader
from forte.pipeline import Pipeline
from forte.data.data_pack import DataPack
from forte.data.extractor import \
    AttributeExtractor, CharExtractor, LinkExtractor, BaseExtractor
from forte.trainer.base.trainer import BaseTrainer
from forte.train_preprocessor import TrainPreprocessor
from ft.onto.base_ontology import Token, Sentence, PredicateLink
from forte.models.srl_new import data

logger = logging.getLogger(__name__)


class SRLTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()
        self.num_epochs = 5
        self.lr = 0.01
        self.momentum = 0.9
        self.nesterov = True
        self.batch_size = 64

        self.train_path = "data/train/"
        self.val_path = "data/dev/"

    def create_tp_request(self) -> Dict:
        text_extractor: AttributeExtractor = \
            AttributeExtractor(config={"entry_type": Token,
                                       "attribute": "text",
                                       "vocab_method": "indexing"})

        char_extractor: CharExtractor = \
            CharExtractor(config={"entry_type": Token,
                                  "vocab_method": "indexing"})

        raw_extractor: AttributeExtractor = \
            AttributeExtractor(config={"entry_type": Token,
                                       "attribute": "text",
                                       "vocab_method": "raw",
                                       "need_pad": False})

        link_extractor: LinkExtractor = \
            LinkExtractor(config={
                "entry_type": PredicateLink,
                "attribute": "arg_type",
                "tagging_unit": Token,
                "vocab_method": "indexing",
                "need_pad": False
            })

        raw_converter: Converter = Converter({"to_numpy": False,
                                              "to_torch": False})

        link_converter: Converter = Converter({"to_numpy": False,
                                               "to_torch": False})

        tp_request: Dict = {
            "scope": Sentence,
            "schemes": {
                "text_tag": {
                    "type": TrainPreprocessor.DATA_INPUT,
                    "extractor": text_extractor
                },
                "char_tag": {
                    "type": TrainPreprocessor.DATA_INPUT,
                    "extractor": char_extractor
                },
                "raw_text_tag": {
                    "type": TrainPreprocessor.DATA_INPUT,
                    "extractor": raw_extractor,
                    "converter": raw_converter
                },
                "pred_link_tag": {  # predicate link
                    "type": TrainPreprocessor.DATA_OUTPUT,
                    "extractor": link_extractor,
                    "converter": link_converter
                }
            }
        }

        return tp_request

    def create_tp_config(self) -> Dict:
        tp_config: Dict = {
            "dataset": {
                "batch_size": self.batch_size
            }
        }

        return tp_config

    def create_pack_iterator(self) -> Iterator[DataPack]:
        srl_train_reader = OntonotesReader(cache_in_memory=True)
        train_pl: Pipeline = Pipeline()
        train_pl.set_reader(srl_train_reader)
        train_pl.initialize()
        pack_iterator = train_pl.process_dataset(self.train_path)

        return pack_iterator

    def train(self):
        def predict_forward_fn(_model: LabeledSpanGraphNetwork,
                               _batch: Dict) -> Dict:
            model_output: List[Dict[int, List[data.Span]]] = \
                _model.decode(text=_batch["raw_text_tag"]["data"],
                              char_batch=_batch["char_tag"]["data"],
                              char_masks=_batch["char_tag"]["masks"],
                              text_batch=_batch["text_tag"]["data"],
                              text_mask=_batch["text_tag"]["masks"][0])

            link_output: List[Dict] = []
            for model_output_i in model_output:
                output_i: Dict = {
                    "data": [],
                    "parent_unit_span": [],
                    "child_unit_span": []
                }
                spans: List[data.Span]
                for predicate_id, spans in model_output_i.items():
                    for span in spans:
                        output_i["data"].append(span.label)
                        output_i["parent_unit_span"].append(
                            (predicate_id, predicate_id + 1))
                        output_i['child_unit_span'].append(
                            (span.start, span.end))
                link_output.append(output_i)

            return {'pred_link_tag': link_output}

        schemes: Dict = self.train_preprocessor.request["schemes"]
        text_extractor: BaseExtractor = schemes["text_tag"]["extractor"]
        char_extractor: BaseExtractor = schemes["char_tag"]["extractor"]
        link_extractor: BaseExtractor = schemes["pred_link_tag"]["extractor"]

        self.model: LabeledSpanGraphNetwork = \
            LabeledSpanGraphNetwork(word_vocab=text_extractor.get_dict(),
                                    char_vocab_size=char_extractor.size(),
                                    label_vocab=link_extractor.get_dict())

        optim: Optimizer = SGD(self.model.parameters(),
                               lr=self.lr,
                               momentum=self.momentum,
                               nesterov=self.nesterov)

        srl_val_reader = OntonotesReader()
        predictor = Predictor(
            batch_size=self.batch_size,
            model=self.model,
            predict_forward_fn=predict_forward_fn,
            feature_resource=self.train_preprocessor.request,
            cross_pack=False)
        val_pl: Pipeline = Pipeline()
        val_pl.set_reader(srl_val_reader)
        val_pl.add(predictor)
        # TODO: We need an evaluator here for SRL task

        logger.info("Start training.")
        epoch = 0
        train_loss: float = 0.0
        train_total: int = 0

        while epoch < self.num_epochs:
            epoch += 1

            # Get iterator of preprocessed batch of train data
            train_batch_iter: Iterator[Batch] = \
                self.train_preprocessor.get_train_batch_iterator()

            for batch in train_batch_iter:
                char_tensor: Tensor = batch["char_tag"]["data"]
                char_masks: List[Tensor] = batch["char_tag"]["masks"]
                text_tensor: Tensor = batch["text_tag"]["data"]
                text_mask: Tensor = batch["text_tag"]["masks"][0]
                text: List[List[str]] = batch["raw_text_tag"]["data"]
                pred_link_features: List[Feature] = batch["pred_link_tag"][
                    "features"]

                optim.zero_grad()

                output: LabeledSpanGraphNetwork.ReturnType = \
                    self.model(text=text,
                               char_batch=char_tensor,
                               char_masks=char_masks,
                               text_batch=text_tensor,
                               text_mask=text_mask,
                               srl_features=pred_link_features)

                output["loss"].backward()
                optim.step()

                train_loss += output["loss"].item()
                train_total += 1

            logger.info("%dth Epoch training, "
                        "loss: %f", epoch, train_loss / train_total)

            train_loss = 0.0
            train_total = 0

            val_pl.run(self.val_path)

            logger.info("%dth Epoch evaluating", epoch)
