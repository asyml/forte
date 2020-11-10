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
import argparse
import os
import yaml

from forte.common.configuration import Config
from forte.data.readers import MSMarcoPassageReader
from examples.data_augmentation import CreateIndexerPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="./config.yml",
                        help="Config YAML filepath")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_file, "r"))
    config = Config(config, default_hparams=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             config.data.relative_path)

    reader = MSMarcoPassageReader()
    indexer = CreateIndexerPipeline(reader=reader, reader_config=config.reader)
    indexer.create_index(data_path)
