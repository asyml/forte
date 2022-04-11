# Copyright 2019 The Forte Authors. All Rights Reserved.
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
import os
from typing import Dict, List, Tuple, Optional

try:
    import texar.torch as tx
except ImportError as e:
    raise ImportError(
        " `texar-pytorch` is not installed correctly."
        " Consider install texar via `pip install texar-pytorch`"
        " Or refer to [extra requirement for texar models](pip install forte[nlp])"
        " for more information."
    ) from e


import torch

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.data.span import Span
from forte.models.srl.model import LabeledSpanGraphNetwork
from forte.processors.base.batch_processor import RequestPackingProcessor
from ft.onto.base_ontology import (
    PredicateLink,
    PredicateMention,
    PredicateArgument,
)

logger = logging.getLogger(__name__)

__all__ = [
    "SRLPredictor",
]

Prediction = List[Tuple[Span, List[Tuple[Span, str]]]]


class SRLPredictor(RequestPackingProcessor):
    """
    An Semantic Role labeler trained according to `He, Luheng, et al.
    "Jointly predicting predicates and arguments in neural semantic role
    labeling." <https://aclweb.org/anthology/P18-2058>`_.
    """

    word_vocab: tx.data.Vocab
    char_vocab: tx.data.Vocab
    model: LabeledSpanGraphNetwork

    def __init__(self):
        super().__init__()
        self.device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

    def initialize(self, resources: Resources, configs: Optional[Config]):
        super().initialize(resources, configs)

        model_dir = configs.storage_path if configs is not None else None
        logger.info("restoring SRL model from %s", model_dir)

        # initialize the batcher
        if configs:
            self.batcher.initialize(configs.batcher)

        self.word_vocab = tx.data.Vocab(
            os.path.join(model_dir, "embeddings/word_vocab.english.txt")
        )
        self.char_vocab = tx.data.Vocab(
            os.path.join(model_dir, "embeddings/char_vocab.english.txt")
        )
        model_hparams = LabeledSpanGraphNetwork.default_hparams()
        model_hparams["context_embeddings"]["path"] = os.path.join(
            model_dir, model_hparams["context_embeddings"]["path"]
        )
        model_hparams["head_embeddings"]["path"] = os.path.join(
            model_dir, model_hparams["head_embeddings"]["path"]
        )
        self.model = LabeledSpanGraphNetwork(
            self.word_vocab, self.char_vocab, model_hparams
        )
        self.model.load_state_dict(
            torch.load(
                os.path.join(model_dir, "pretrained/model.pt"),
                map_location=self.device,
            )
        )
        self.model.eval()

    def predict(self, data_batch: Dict) -> Dict[str, List[Prediction]]:
        text: List[List[str]] = [
            sentence.tolist() for sentence in data_batch["Token"]["text"]
        ]
        text_ids, length = tx.data.padded_batch(
            [
                self.word_vocab.map_tokens_to_ids_py(sentence)
                for sentence in text
            ]
        )
        text_ids = torch.from_numpy(text_ids).to(device=self.device)
        length = torch.tensor(length, dtype=torch.long, device=self.device)
        batch_size = len(text)
        batch = tx.data.Batch(
            batch_size,
            text=text,
            text_ids=text_ids,
            length=length,
            srl=[[]] * batch_size,
        )
        self.model = self.model.to(self.device)
        batch_srl_spans = self.model.decode(batch)

        # Convert predictions into annotations.
        batch_predictions: List[Prediction] = []
        for idx, srl_spans in enumerate(batch_srl_spans):
            word_spans = data_batch["Token"]["span"][idx]
            predictions: Prediction = []
            for pred_idx, pred_args in srl_spans.items():
                begin, end = word_spans[pred_idx]
                # TODO cannot create annotation here.
                # Need to convert from Numpy numbers to int.
                pred_span = Span(begin.item(), end.item())
                arguments = []
                for arg in pred_args:
                    begin = word_spans[arg.start][0].item()
                    end = word_spans[arg.end][1].item()
                    arg_annotation = Span(begin, end)
                    arguments.append((arg_annotation, arg.label))
                predictions.append((pred_span, arguments))
            batch_predictions.append(predictions)
        return {"predictions": batch_predictions}

    def pack(
        self,
        pack: DataPack,
        predict_results: Dict[str, List[Prediction]],
        _: Optional[Annotation] = None,
    ):
        batch_predictions = predict_results["predictions"]
        for predictions in batch_predictions:
            for pred_span, arg_result in predictions:

                pred = PredicateMention(pack, pred_span.begin, pred_span.end)

                for arg_span, label in arg_result:
                    arg = PredicateArgument(pack, arg_span.begin, arg_span.end)
                    link = PredicateLink(pack, pred, arg)
                    link.arg_type = label

    @classmethod
    def default_configs(cls):
        """
        This defines the default configuration structure for the predictor.
        """
        return {
            "storage_path": None,
            "batcher": {
                "batch_size": 4,
                "context_type": "ft.onto.base_ontology.Sentence",
                "requests": {"ft.onto.base_ontology.Token": []},
            },
        }
