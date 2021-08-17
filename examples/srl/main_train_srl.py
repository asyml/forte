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
import torch
import yaml

from srl_trainer import SRLTrainer

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    trainer: SRLTrainer = SRLTrainer(
        config_extractors=yaml.safe_load(
            open("configs/config_extractors.yml", "r")
        ),
    )

    trainer.run()

    # Save training state to disk
    trainer.save("train_state.pt")
    torch.save(trainer.model, "model.pt")
