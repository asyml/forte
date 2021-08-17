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
from pathlib import Path

import yaml

from forte.common.configuration import Config
from forte.data.data_utils import maybe_download


def _get_default_config():
    return {"relative_path": "./data/collectionandqueries"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data path can be read from config file
    parser.add_argument(
        "--config_file", default="./config.yml", help="Config YAML filepath"
    )
    args = parser.parse_args()

    data_config = yaml.safe_load(open(args.config_file, "r"))["data"]
    config = Config(data_config, default_hparams=_get_default_config())

    # data path can be passed through command line, it is given priority
    default_data_path = config.relative_path
    parser.add_argument(
        "--path",
        default=default_data_path,
        help="Path to where data will be saved",
    )

    args = parser.parse_args()

    resource_path = Path(args.path)

    # create the path if it doesn't exist
    resource_path.mkdir(parents=True, exist_ok=True)

    # download data
    url = (
        "https://msmarco.blob.core.windows.net/msmarcoranking/"
        "collectionandqueries.tar.gz"
    )

    maybe_download(urls=[url], path=resource_path, extract=True)
