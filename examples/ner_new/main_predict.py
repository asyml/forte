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

import yaml

from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from forte.data.extractor.predictor import Predictor
from ft.onto.base_ontology import Token, Sentence, EntityMention

config_data = yaml.safe_load(open("configs/config_data.yml", "r"))
config_predict = yaml.safe_load(open("configs/config_predict.yml", "r"))

config = Config({}, default_hparams=None)
config.add_hparam("config_data", config_data)

pl = Pipeline[DataPack]()
pl.set_reader(CoNLL03Reader())
pl.add(Predictor())

pl.initialize()

for pack in pl.process_dataset(config.config_data.test_path):
    for instance in pack.get(Sentence):
        sent = instance.text
        ner_tags = []
        for entry in pack.get(EntityMention, instance):
            ner_tags.append((entry.text, entry.ner_type))
        print('---------')
        print(sent)
        print(ner_tags)

