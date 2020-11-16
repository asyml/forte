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
import torch

from examples.ner_new.ner_model_processor import NerModelProcessor
import yaml

from forte.data.types import DATA_INPUT, DATA_OUTPUT
from forte.common.configuration import Config
from forte.data.extractor.extractor import \
    BioSeqTaggingExtractor, TextExtractor, CharExtractor
from forte.data.extractor.train_pipeline import TrainPipeline
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from ft.onto.base_ontology import Sentence, Token, EntityMention

# All the configs
config_data = yaml.safe_load(open("configs/config_data.yml", "r"))
config_model = yaml.safe_load(open("configs/config_model.yml", "r"))
config_preprocess = \
    yaml.safe_load(open("configs/config_preprocessor.yml", "r"))

config = Config({}, default_hparams=None)
config.add_hparam('config_data', config_data)
config.add_hparam('config_model', config_model)
config.add_hparam('preprocessor', config_preprocess)

request = {
    "scope": Sentence,
    "schemes": {
        "text_tag": {
            "entry_type": Token,
            "repr": "text_repr",
            "conversion_method": "indexing",
            "vocab_use_pad": True,
            "type": DATA_INPUT,
            "extractor": TextExtractor
        },
        "char_tag": {
            "entry_type": Token,
            "repr": "char_repr",
            "conversion_method": "indexing",
            "max_char_length": config.config_data.max_char_length,
            "vocab_use_pad": True,
            "type": DATA_INPUT,
            "extractor": CharExtractor
        },
        "ner_tag": {
            "entry_type": EntityMention,
            "attribute": "ner_type",
            "based_on": Token,
            "strategy": "BIO",
            "vocab_method": "indexing",
            "vocab_use_pad": True,
            "type": DATA_OUTPUT,
            "extractor": BioSeqTaggingExtractor
        }
    }
}

# All not specified dataset parameters are set by default in Texar.
# Default settings can be found here:
# https://texar-pytorch.readthedocs.io/en/latest/code/data.html#texar.torch.data.DatasetBase.default_hparams
config = {
    "data_pack": {
        "train_loader": {
            "src_dir": config.config_data.train_path
        },
        "val_loader": {
            "src_dir": config.config_data.val_path
        }
    },
    "train": {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_epochs": config.config_data.num_epochs
    },
    "dataset": {
        "batch_size": config.config_data.batch_size_tokens
    }
}

ner_reader = CoNLL03Reader()
ner_model_processor = NerModelProcessor()

train_pipeline = \
    TrainPipeline(train_reader=ner_reader,
                  val_reader=ner_reader,
                  model_processor=ner_model_processor,
                  request=request,
                  config=config)

train_pipeline.run()

# Save training result to disk
# train_pipeline.save_state(config.config_data.train_state_path)
# torch.save(ner_model_processor, config.config_model.model_path)
