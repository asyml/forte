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

"""
This example script shows how Forte design the training process to split
models away from data processing using the extractor interface.

In this script, the first positional argument can be either "ner" or "pos",
which determines what task is to be trained. Only the extractor part need to
be changed to make this repurposing happen, demonstrating that extractor can
be used as an interface that separated data logic form model logic.

To run this script, do:

python main_train_tagging pos

or

python main_train_tagging ner
"""

import logging
import sys
import torch
import yaml

from tagging_trainer import TaggingTrainer

from forte.common.configuration import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    task = sys.argv[1]
    assert task in ["ner", "pos"], "Not supported nlp task type: {}".format(
        task
    )

    extractor_configs = yaml.safe_load(
        open("configs/config_extractors.yml", "r")
    )

    # Configure output extractor based on the task, see
    #   BioSeqTaggingExtractor for more information.
    output_configs = extractor_configs["feature_scheme"]["output_tag"][
        "extractor"
    ]["config"]
    if task == "ner":
        output_configs["entry_type"] = "ft.onto.base_ontology.EntityMention"
        output_configs["attribute"] = "ner_type"
        output_configs["tagging_unit"] = "ft.onto.base_ontology.Token"
    elif task == "pos":
        output_configs["entry_type"] = "ft.onto.base_ontology.Token"
        output_configs["attribute"] = "pos"

    config = {
        "config_data": Config(
            {},
            default_hparams=yaml.safe_load(
                open("configs/config_data.yml", "r")
            ),
        ),
        "config_model": Config(
            {},
            default_hparams=yaml.safe_load(
                open("configs/config_model.yml", "r")
            ),
        ),
        "config_extractors": extractor_configs,
        "device": torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    }

    trainer: TaggingTrainer = TaggingTrainer(task_type=task, **config)
    trainer.run()

    # Save training state to disk
    trainer.save(config["config_data"].train_state_path)
    torch.save(trainer.model, "model.pt")
