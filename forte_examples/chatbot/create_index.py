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
import argparse
import pickle
import importlib
import logging
import yaml

import torch
import texar.torch as tx

from forte.common.configuration import Config
from forte.indexers.embedding_based_indexer import EmbeddingBasedIndexer

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--config_data", default="config_data",
                    help="File to read the config from")
args = parser.parse_args()

config = yaml.safe_load(open("config.yml", "r"))
config = Config(config, default_hparams=None)
config_data = importlib.import_module(args.config_data)


class Indexer:

    def __init__(self, model_path, torch_device=None):

        self.bert = tx.modules.BERTEncoder(
            pretrained_model_name=None, hparams={"pretrained_model_name": None})
        self.device = torch_device
        self.bert.to(device=self.device)

        with open(model_path, "rb") as f:
            state_dict = pickle.load(f)

        self.bert.load_state_dict(state_dict["bert"])

        self.tokenizer = tx.data.BERTTokenizer(
            pretrained_model_name="bert-base-uncased")

        self.index = EmbeddingBasedIndexer(config={
            "index_type": "GpuIndexFlatIP", "dim": 768, "device": "gpu0"})

    @torch.no_grad()
    def create_index(self):
        hparams = {
            "allow_smaller_final_batch": True,
            "batch_size": 128,
            "dataset": {
                "data_name": "data",
                "feature_types": config_data.feature_types,
                "files": ["data/train.pkl", "data/eval.pkl", "data/test.pkl"]
            },
            "shuffle": False
        }

        dataset = tx.data.RecordData(hparams=hparams, device=self.device)
        data_iterator = tx.data.DataIterator(dataset)

        start = 0
        for idx, batch in enumerate(data_iterator):
            ids = range(start, start + len(batch))
            text = batch["sentence_b"]
            output, _ = self.bert(inputs=batch["sent_b_input_ids"],
                                  sequence_length=batch["sent_b_seq_len"],
                                  segment_ids=batch["sent_b_segment_ids"])
            cls_tokens = output[:, 0, :]  # CLS token is first token
            self.index.add(vectors=cls_tokens, meta_data=dict(zip(ids, text)))

            start += len(batch)

            if (idx + 1) % 50 == 0:
                logging.info("Completed %s batches of size %s", idx + 1,
                             config.indexer.batch_size)

        self.index.save(path=config.indexer.model_dir)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indexer = Indexer(model_path="model/chatbot_model.ckpt",
                      torch_device=device)
    indexer.create_index()
