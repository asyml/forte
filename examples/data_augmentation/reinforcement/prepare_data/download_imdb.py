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

import os
import sys
import subprocess


def main():
    if not os.path.exists("data/IMDB_raw"):
        subprocess.run("mkdir data/IMDB_raw", shell=True, check=True)
    # pylint: disable=line-too-long
    subprocess.run(
        'wget -P data/IMDB_raw/ https://github.com/google-research/uda/blob/master/text/data/IMDB_raw/train_id_list.txt',
        shell=True, check=True)
    subprocess.run(
        'wget -P data/IMDB_raw/ https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
        shell=True, check=True)
    subprocess.run(
        'tar xzvf data/IMDB_raw/aclImdb_v1.tar.gz -C data/IMDB_raw/ && rm data/IMDB_raw/aclImdb_v1.tar.gz',
        shell=True, check=True)


if __name__ == '__main__':
    sys.exit(main())
