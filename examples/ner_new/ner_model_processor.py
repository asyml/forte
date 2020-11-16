#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

# All the configs
import logging
import os
from pathlib import Path
import numpy as np
import torch
import yaml
from typing import Dict, Optional, List

from ft.onto.base_ontology import Sentence, EntityMention, Token
from texar.torch.data import Batch
from torch.optim import SGD
from torch import nn
from torch.optim.optimizer import Optimizer

from forte.data.base_pack import PackType
from forte.data.extractor.model_processor import ModelProcessor
from forte.models.ner.model_factory import BiRecurrentConvCRF
from forte.models.ner.utils import load_glove_embedding
from forte.data.extractor.extractor import BaseExtractor
from forte.common.configuration import Config


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


def _write_tokens_to_file(pred_pack, pred_request,
                          refer_pack, refer_request,
                          output_filename):
    opened_file = open(output_filename, "w+")
    for pred_sentence, tgt_sentence in zip(
            pred_pack.get_data(**pred_request),
            refer_pack.get_data(**refer_request)
    ):

        pred_entity_mention, tgt_entity_mention = \
            pred_sentence["EntityMention"], tgt_sentence["EntityMention"]
        tgt_tokens = tgt_sentence["Token"]

        tgt_ptr, pred_ptr = 0, 0

        for i in range(len(tgt_tokens["text"])):
            w = tgt_tokens["text"][i]
            p = tgt_tokens["pos"][i]
            ch = tgt_tokens["chunk"][i]
            # TODO: This is not correct and probably we need a utility to do
            #       BIO encoding to get ner_type?
            if tgt_ptr < len(tgt_entity_mention["span"]) and \
                    (tgt_entity_mention["span"][tgt_ptr] ==
                     tgt_tokens["span"][i]).all():
                tgt = tgt_entity_mention["ner_type"][tgt_ptr]
                tgt_ptr += 1
            else:
                tgt = "O"

            if pred_ptr < len(pred_entity_mention["span"]) and \
                    (pred_entity_mention["span"][pred_ptr] ==
                     tgt_tokens["span"][i]).all():
                pred = pred_entity_mention["ner_type"][pred_ptr]
                pred_ptr += 1
            else:
                pred = "O"

            opened_file.write(
                "%d %s %s %s %s %s\n" % (i + 1, w, p, ch, tgt, pred)
            )

        opened_file.write("\n")
    opened_file.close()


class NerModelProcessor(ModelProcessor):
    def __init__(self):
        super().__init__()

        # Load user-defined configuration
        self.config_model = \
            yaml.safe_load(open("configs/config_model.yml", "r"))
        self.config_preprocess = \
            yaml.safe_load(open("configs/config_preprocessor.yml", "r"))

        self.config = Config({}, default_hparams=None)
        self.config.add_hparam('config_model', self.config_model)
        self.config.add_hparam('preprocessor', self.config_preprocess)

        self.model: Optional[nn.Module] = None
        self.optim: Optional[Optimizer] = None

        self.train_err: float = 0.0
        self.train_total: float = 0.0
        self.val_scores: List = []
        self.output_file = "tmp_eval.txt"
        self.score_file = "tmp_eval.score"
        self.scores: Dict[str, float] = {}

        self.logger = logging.getLogger(__name__)

    def _create_model(self,
                      schemes: Dict[str, Dict[str, BaseExtractor]],
                      pl_config: Config):
        text_extractor: BaseExtractor = schemes["text_tag"]["extractor"]
        char_extractor: BaseExtractor = schemes["char_tag"]["extractor"]
        ner_extractor: BaseExtractor = schemes["ner_tag"]["extractor"]

        # embedding_dict = \
        #     load_glove_embedding(config.preprocessor.embedding_path)
        #
        # for word in embedding_dict:
        #     if not text_extractor.contains(word):
        #         text_extractor.add_entry(word)
        #

        # TODO: temporarily make fake pretrained emb for debugging
        embedding_dict = {}
        fake_tensor = torch.tensor([0.0 for i in range(100)])
        for word, index in text_extractor.items():
            embedding_dict[word] = fake_tensor

        word_embedding_table = \
            construct_word_embedding_table(embedding_dict, text_extractor)

        model: nn.Module = \
            BiRecurrentConvCRF(word_embedding_table=word_embedding_table,
                               char_vocab_size=char_extractor.size(),
                               tag_vocab_size=ner_extractor.size(),
                               config_model=self.config.config_model)
        model.to(device=pl_config.train.device)

        return model

    def setup(self,
              schemes: Dict[str, Dict[str, BaseExtractor]],
              pl_config: Config):
        # self.config: user-defined configs
        # pl_config: Forte train pipeline configs

        self.model: nn.Module = self._create_model(schemes, pl_config)
        self.optim: Optimizer = SGD(
            self.model.parameters(),
            lr=self.config.config_model.learning_rate,
            momentum=self.config.config_model.momentum,
            nesterov=True)
        self.train_err = 0.0
        self.train_total = 0.0
        self.val_scores = []

    def train(self, batch: Batch):
        word = batch["text_tag"]["tensor"]
        char = batch["char_tag"]["tensor"]
        ner = batch["ner_tag"]["tensor"]
        word_masks = batch["text_tag"]["mask"][0]

        self.optim.zero_grad()

        loss = self.model(word, char, ner, mask=word_masks)

        loss.backward()
        self.optim.step()

        batch_train_err = loss.item() * batch.batch_size

        self.train_err += batch_train_err
        self.train_total += batch.batch_size

    def train_finish(self, epoch: int):
        self.logger.info("%dth Epoch training, "
                         "train error rate = %f",
                         epoch, self.train_err / self.train_total)

        self.train_err = 0.0
        self.train_total = 0.0

    def predict(self, batch: Dict) -> Dict:
        word = batch["text_tag"]["tensor"]
        char = batch["char_tag"]["tensor"]
        word_masks = batch["text_tag"]["mask"][0]

        output = self.model.decode(input_word=word,
                                   input_char=char,
                                   mask=word_masks)
        output = output.numpy()
        return {'ner_tag': output}

    def evaluate(self, pred_pack: PackType, ref_pack: PackType):
        pred_getdata_args = {
            "context_type": Sentence,
            "request": {
                EntityMention: {
                    "fields": ["ner_type"],
                },
                Sentence: [],  # span by default
            },
        }

        refer_getdata_args = {
            "context_type": Sentence,
            "request": {
                Token: {
                    "fields": ["chunk", "pos", "ner"]
                },
                EntityMention: {
                    "fields": ["ner_type"],
                },
                Sentence: [],  # span by default
            }
        }

        _write_tokens_to_file(pred_pack=pred_pack,
                              pred_request=pred_getdata_args,
                              refer_pack=ref_pack,
                              refer_request=refer_getdata_args,
                              output_filename=self.output_file)
        eval_script = \
            Path(os.path.abspath(__file__)).parents[2] / \
            "forte/utils/eval_scripts/conll03eval.v2"
        os.system(f"perl {eval_script} < {self.output_file} > "
                  f"{self.score_file}")
        with open(self.score_file, "r") as fin:
            fin.readline()
            line = fin.readline()
            fields = line.split(";")
            acc = float(fields[0].split(":")[1].strip()[:-1])
            precision = float(fields[1].split(":")[1].strip()[:-1])
            recall = float(fields[2].split(":")[1].strip()[:-1])
            f1 = float(fields[3].split(":")[1].strip())

        val_score = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        self.val_scores.append(val_score)

    def evaluate_finish(self, epoch: int):
        self.logger.info("%dth Epoch evaluate, "
                         "each pack evaluate score:", epoch)

        for id, val_score in enumerate(self.val_scores):
            self.logger.info("pack %d, val score = %s",
                             id, val_score)

        self.val_scores = []
