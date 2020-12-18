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
"""This file predict the ner tag for conll03 dataset."""
import sys
import yaml
import torch
from forte.pipeline import Pipeline
from forte.data.readers.conll03_reader_new import CoNLL03Reader
from forte.predictor import Predictor
from ft.onto.base_ontology import Sentence, EntityMention, Token
from examples.tagging.evaluator import CoNLLNEREvaluator


def predict_forward_fn(model, batch):
    '''Use model and batch data to predict ner tag.'''
    word = batch["text_tag"]["data"]
    char = batch["char_tag"]["data"]
    word_masks = batch["text_tag"]["masks"][0]
    output = model.decode(input_word=word, input_char=char, mask=word_masks)
    output = output.numpy()
    return {'output_tag': output}


task = sys.argv[1]
assert task in ["ner", "pos"], \
    "Not supported nlp task type: {}".format(task)

config_predict = yaml.safe_load(open("configs/config_predict.yml", "r"))
saved_model = torch.load(config_predict['model_path'])
train_state = torch.load(config_predict['train_state_path'])

reader = CoNLL03Reader()
predictor = Predictor(batch_size=config_predict['batch_size'],
                model=saved_model,
                predict_forward_fn=predict_forward_fn,
                feature_resource=train_state['feature_resource'])
evaluator = CoNLLNEREvaluator()


pl = Pipeline()
pl.set_reader(reader)
pl.add(predictor)
pl.add(evaluator)
pl.initialize()


for pack in pl.process_dataset(config_predict['test_path']):
    print("---- pack ----")
    for instance in pack.get(Sentence):
        sent = instance.text
        output_tags = []
        if task == "ner":
            for entry in pack.get(EntityMention, instance):
                output_tags.append((entry.text, entry.ner_type))
        else:
            for entry in pack.get(Token, instance):
                output_tags.append((entry.text, entry.pos))
        print('---- example -----')
        print("sentence: ", sent)
        print("output_tags: ", output_tags)
    print(evaluator.get_result())
