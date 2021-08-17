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
"""Predict for SRL task."""
from typing import List
import torch
from torch import Tensor
from ft.onto.base_ontology import Sentence
from forte.data.converter.feature import Feature
from forte.pipeline import Pipeline
from forte.data.readers.ontonotes_reader import OntonotesReader
from forte.predictor import Predictor


reader = OntonotesReader()
saved_model = torch.load("model.pt")


def predict_forward_fn(model, batch):
    '''Predict function.'''
    char_tensor: Tensor = batch["char_tag"]["tensor"]
    char_masks: List[Tensor] = batch["char_tag"]["mask"]
    text_tensor: Tensor = batch["text_tag"]["tensor"]
    text_mask: Tensor = batch["text_tag"]["mask"][0]
    raw_text_features: List[Feature] = batch["raw_text_tag"]["features"]
    srl_features: List[Feature] = [None] * len(raw_text_features)

    text: List[List[str]] = []
    for feature in raw_text_features:
        text.append(feature.unroll()[0])

    output = model.decode(text=text,
                          char_batch=char_tensor,
                          char_masks=char_masks,
                          text_batch=text_tensor,
                          text_mask=text_mask,
                          srl_features=srl_features)
    print(output)
    return {"pred_link_tag": output}


train_state = torch.load("train_state.pkl")

predictor = Predictor(batch_size=10,
                        model=saved_model,
                        predict_forward_fn=predict_forward_fn,
                        feature_resource=train_state["feature_resource"])

pl = Pipeline()
pl.set_reader(reader)
pl.add(predictor)
pl.initialize()

for pack in pl.process_dataset("data/test"):
    print("====== pack ======")
    for instance in pack.get(Sentence):
        pass
