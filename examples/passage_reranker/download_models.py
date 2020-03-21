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

from forte.data.data_utils import maybe_download

# download resources
urls = ["https://drive.google.com/file/d/1j3i5U1YffYqKTdSbnlsrMAm9j86nLjxC/view"
        "?usp=sharing",
        "https://drive.google.com/file/d/1aRPS_b4AFaZTsk9uZ41tkWIBvWbO_s_V/"
        "view?usp=sharing",
        "https://drive.google.com/file/d/1SYpcWEDeTlbOsXlRevS8YS_dyP_k29g0/"
        "view?usp=sharing",
        "https://drive.google.com/file/d/1S2UMDBX7Ci-Mrm30434t0LOBL__Db92Y/"
        "view?usp=sharing",
        "https://drive.google.com/file/d/1O4iFhBPuogwEgz7bpJjEqDqAlYf5caP4/"
        "view?usp=sharing"]

filenames = ["model.pkl", "word_embedding_table.pkl", "word_alphabet.pkl",
             "ner_alphabet.pkl", "char_alphabet.pkl"]

maybe_download(urls=urls, path="resources/", filenames=filenames)
