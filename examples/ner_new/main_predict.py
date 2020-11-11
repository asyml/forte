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
import torch
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from forte.data.extractor.predictor import Predictor
from ft.onto.base_ontology import Token, Sentence, EntityMention
from forte.data.extractor.unpadder import SameLengthUnpadder


config_predict = yaml.safe_load(open("configs/config_predict.yml", "r"))


pretrain_model = torch.load(config_predict['model_path'])

def predict_forward_fn(model, batch):
    word = batch["text_tag"]["tensor"]
    char = batch["char_tag"]["tensor"]
    word_masks = batch["text_tag"]["mask"][0]
    output = model.decode(input_word=word, input_char=char, mask=word_masks)
    output = output.numpy()
    return {'ner_tag': output}


train_state = torch.load(config_predict['train_state_path'])

pl = Pipeline()
pl.set_reader(CoNLL03Reader())
pl.add(Predictor(batch_size=config_predict['batch_size'],
                predict_foward_fn=lambda x: predict_forward_fn(pretrain_model, x),
                feature_resource=train_state['feature_resource']))
pl.initialize()

for pack in pl.process_dataset(config_predict['test_path']):
    for instance in pack.get(Sentence):
        sent = instance.text
        ner_tags = []
        for entry in pack.get(EntityMention, instance):
            ner_tags.append((entry.text, entry.ner_type))
        print('---------')
        print(sent)
        print(ner_tags)
