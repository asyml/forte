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


"""
Download IMDB dataset.
"""
from forte.data.data_utils import maybe_download


def main():
    download_path = "data/IMDB_raw"
    maybe_download(urls=[
        "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"],
        path=download_path,
        extract=True)
    maybe_download(urls=[
        "https://storage.googleapis.com/uda_model/text/back_trans_checkpoints.zip"],
        path=download_path,
        extract=True)


if __name__ == '__main__':
    main()
