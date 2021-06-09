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
from collections import OrderedDict
from pathlib import Path
import os

from forte.data.data_utils import maybe_download

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-name", default="srl", help="Model name to download"
)
parser.add_argument(
    "--path", default="model/", help="Path to where the models will be saved"
)
args = parser.parse_args()

if __name__ == "__main__":
    model_path = Path(args.path)

    # create the model path if it doesn't exist
    model_path.mkdir(parents=True, exist_ok=True)

    # download srl model
    if args.model_name.lower() == "srl":
        embedding_path = Path("srl/embeddings")
        (model_path / embedding_path).mkdir(parents=True, exist_ok=True)
        pretrained_path = Path("srl/pretrained")
        (model_path / pretrained_path).mkdir(parents=True, exist_ok=True)

        urls_to_file_names = OrderedDict(
            {
                "https://drive.google.com/file/d/102YRcdXqDFLOjToR7L-3XYcU-yqcKAO8/"
                "view?usp=sharing": os.path.join(
                    str(embedding_path), "char_vocab.english.txt"
                ),
                "https://drive.google.com/file/d/1hgwmUBk8Mb3iZYiHi1UpCpPFOCfOQLLB/"
                "view?usp=sharing": os.path.join(
                    str(embedding_path), "glove.840B.300d.05.filtered"
                ),
                "https://drive.google.com/file/d/1H4PZhJhGoFBqrSMRPufjJ-9zwROw8hAK/"
                "view?usp=sharing": os.path.join(
                    str(embedding_path), "glove_50_300_2.filtered"
                ),
                "https://drive.google.com/file/d/1uoA5EnZMWl5m5DMevGcI7UjiXxQRlD9W/"
                "view?usp=sharing": os.path.join(
                    str(embedding_path), "word_vocab.english.txt"
                ),
                "https://drive.google.com/file/d/1UZc8x-mhdXg7Rtt6FSBDlEoJb_nHxDAQ/"
                "view?usp=sharing": os.path.join(
                    str(pretrained_path), "model.pt"
                ),
            }
        )
        maybe_download(
            urls=list(urls_to_file_names.keys()),
            path=model_path,
            filenames=list(urls_to_file_names.values()),
        )

    # download indexer model
    elif args.model_name.lower() == "indexer":
        urls_to_file_names = OrderedDict(
            {
                "https://drive.google.com/file/d/14lL6AoyjdCp-fj8DOlyZhQZrNwoJBfQm/"
                "view?usp=sharing": "index.faiss",
                "https://drive.google.com/file/d/1DdgMA7jttgA113EIlebVb33JpGCGbEYf/"
                "view?usp=sharing": "index.meta_data",
            }
        )
        maybe_download(
            urls=list(urls_to_file_names.keys()),
            path=model_path / "chatbot",
            filenames=list(urls_to_file_names.values()),
        )

    # download chatbot model
    elif args.model_name.lower() == "chatbot-bert":
        maybe_download(
            urls=[
                "https://drive.google.com/file/d/1_DbcrgJ_rRsX9k8i"
                "9zVv7sDiTGrek9-p/view?usp=sharing"
            ],
            path=model_path,
            filenames=["chatbot_model.ckpt"],
        )
    else:
        print(
            f"Incorrect 'model-name' {args.model_name}. Available values are "
            f"'srl' and 'indexer' and 'chatbot-bert'."
        )
