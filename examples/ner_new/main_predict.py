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
from forte.pipeline import Pipeline
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from forte.data.extractor.predictor import Predictor
from ft.onto.base_ontology import Sentence, EntityMention
from examples.ner_new.ner_evaluator import CoNLLNEREvaluator

def predict_forward_fn(model, batch):
    word = batch["text_tag"]["tensor"]
    char = batch["char_tag"]["tensor"]
    word_masks = batch["text_tag"]["mask"][0]
    output = model.decode(input_word=word, input_char=char, mask=word_masks)
    output = output.numpy()
    return {'ner_tag': output}

config_predict = yaml.safe_load(open("configs/config_predict.yml", "r"))
model = torch.load(config_predict['model_path'])
train_state = torch.load(config_predict['train_state_path'])

pl = Pipeline()
pl.set_reader(CoNLL03Reader())
pl.add(Predictor(batch_size=config_predict['batch_size'],
                model = model,
                predict_forward_fn = predict_forward_fn,
                feature_resource = train_state['feature_resource']))
pl.add(CoNLLNEREvaluator())
pl.initialize()


for pack in pl.process_dataset(config_predict['test_path']):
    for instance in pack.get(Sentence):
        sent = instance.text
        ner_tags = []
        for entry in pack.get(EntityMention, instance):
            ner_tags.append((entry.text, entry.ner_type))
        print('---- example -----')
        print("sentence: ", sent)
        print("ner_tags: ", ner_tags)
