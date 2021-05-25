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
import logging
import os

import yaml

from forte_wrapper.elastic import ElasticSearchTextIndexProcessor

from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.readers import MSMarcoPassageReader
from forte.pipeline import Pipeline


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="./config.yml",
                        help="Config YAML filepath")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_file, "r"))
    config = Config(config, default_hparams=None)

    nlp: Pipeline[DataPack] = Pipeline()
    nlp.set_reader(MSMarcoPassageReader())
    nlp.add(ElasticSearchTextIndexProcessor(), config=config.create_index)
    nlp.initialize()

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             config.data.relative_path)

    for idx, pack in enumerate(nlp.process_dataset(data_path)):
        if idx + 1 > 0 and (idx + 1) % 10000 == 0:
            print(f"Indexed {idx + 1} packs")
