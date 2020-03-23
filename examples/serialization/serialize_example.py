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
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers import OntonotesReader
from forte.processors.nltk_processors import NLTKWordTokenizer, \
    NLTKPOSTagger, NLTKSentenceSegmenter
from forte.processors.writers import DocIdJsonPackWriter

nlp = Pipeline[DataPack]()
reader = OntonotesReader()

data_path = "../../data_samples/ontonotes/00/"

nlp.set_reader(OntonotesReader())
nlp.add_processor(NLTKSentenceSegmenter())
nlp.add_processor(NLTKWordTokenizer())
nlp.add_processor(NLTKPOSTagger())

# This is a simple writer that serialize the result to the current directory and
# will use the DocID field in the data pack as the file name.
nlp.add_processor(
    DocIdJsonPackWriter(),
    {
        'output_dir': 'output',
        'indent': 2,
    }
)

nlp.initialize()

nlp.run(data_path)
