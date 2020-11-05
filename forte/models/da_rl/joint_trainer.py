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
import texar.torch as tx
import torch.nn as nn
import random
from tqdm import trange


class DaRlTrainer:
    def __init__(self, classifier, generator, args):
        self.classifier = classifier
        self.generator = generator
        self.args = args

    def pretrain(self, examples):
        # pretrain generator
        for epoch in range(self.args.generator_pretrain_epochs):
            self.generator.train_epoch(examples)

        # pretrain classifier
        for epoch in range(self.args.classifier_pretrain_epochs):
            self.classifier.train_epoch(examples)

    def train_epoch(self, epoch, train_examples):
        random.seed(199 * (epoch + 1))
        random.shuffle(train_examples)

        batch_size = self.args.batch_size
        for i in trange(0, len(train_examples), batch_size, desc='Training'):
            batch_examples = train_examples[i: i + batch_size]
            aug_batch_examples = self.generator.augment(batch_examples)
            all_examples = batch_examples + aug_batch_examples  # append
            self.classifier.train_batch(all_examples)
