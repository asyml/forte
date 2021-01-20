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


from typing import Optional
from forte.common import Resources
from forte.common.configuration import Config
from forte.processors.base.extractor_batch_processor import FixedSizeBatchProcessor


class Predictor(FixedSizeBatchProcessor):
    def initialize(self, resources: Resources, configs: Optional[Config]):
        new_config = {}

        processor_config = {}
        processor_config["scope"] = configs.scope
        processor_config["feature_scheme"] = configs.feature_scheme

        batcher_config = {}
        batcher_config["scope"] = configs.scope
        batcher_config["feature_scheme"] = {}
        for tag, scheme in configs.feature_scheme.items():
            if scheme["type"] == DATA_INPUT:
                batcher_config["feature_scheme"]["tag"] = scheme
        batcher_config["batch_size"] = self.configs.batch_size

        new_config["processor"] = processor_config
        new_config["batcher"] = batcher_config

        super().initialize(resources, configs)

        assert "model" in configs
        self.model = configs.model

    def predict(self, data_batch):
        raise NotImplementedError()
