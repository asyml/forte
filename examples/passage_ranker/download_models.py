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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model name and path can be read from config file
    parser.add_argument(
        "--config_file", default="./config.yml", help="Config YAML filepath"
    )

    args = parser.parse_args()

    config = yaml.safe_load(open(args.config_file, "r"))
    config = Config(config, default_hparams=None)

    default_model_name = config.reranker.pretrained_model_name
    default_model_dir = config.reranker.model_dir

    # model name and path can be passed through command line as well
    parser.add_argument(
        "--pretrained_model_name",
        default=default_model_name,
        help="Pre-trained model name to download. It can be "
        "`bert-base-uncased` or `bert-large-uncased`or a "
        "url from where the model will be directly"
        "downloaded",
    )

    parser.add_argument(
        "--model_dir",
        default=default_model_dir,
        help="Directory to which the model will be downloaded",
    )

    args = parser.parse_args()

    resource_path = Path(args.model_dir)

    # create the path if it doesn't exist
    resource_path.mkdir(parents=True, exist_ok=True)

    # download data
    if args.pretrained_model_name.lower() == "bert-base-uncased":
        gd_id = "1cyUrhs7JaCJTTu-DjFUqP6Bs4f8a6JTX"
        url = f"https://drive.google.com/file/d/{gd_id}/view"

    elif args.pretrained_model_name.lower() == "bert-large-uncased":
        gd_id = "1crlASTMlsihALlkabAQP6JTYIZwC1Wm8"
        url = f"https://drive.google.com/file/d/{gd_id}/view"

    else:
        url = args.pretrained_model_name

    maybe_download(urls=[url], path=resource_path, extract=True)
