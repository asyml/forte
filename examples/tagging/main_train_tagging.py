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
        "config_extractors": yaml.safe_load(
            open("configs/config_extractors.yml", "r")
        ),
        "device": torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    }

    trainer: TaggingTrainer = TaggingTrainer(task_type=task, **config)

    trainer.run()

    # Save training state to disk
    trainer.save(config["config_data"].train_state_path)
    torch.save(trainer.model, "model.pt")
