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

import argparse
import torch
import texar.torch as tx

from forte.models.da_rl import (MetaAugmentationWrapper, DaRlTrainer)

parser = argparse.ArgumentParser()

# data
parser.add_argument("--task", default="sst-5", type=str)
parser.add_argument('--train_num_per_class', default=None, type=int)
parser.add_argument('--dev_num_per_class', default=None, type=int)
parser.add_argument('--data_seed', default=159, type=int)

# train
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--classifier_lr", default=4e-5, type=float)
parser.add_argument("--classifier_pretrain_epochs", default=1, type=int)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--min_epochs", default=0, type=int)

# augmentation
parser.add_argument("--generator_lr", default=4e-5, type=float)
parser.add_argument("--generator_pretrain_epochs", default=60, type=int)
parser.add_argument('--n_aug', default=2, type=int)

args = parser.parse_args()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # prepare data
    examples, label_list = get_data(
        task=args.task,
        train_num_per_class=args.train_num_per_class,
        dev_num_per_class=args.dev_num_per_class,
        data_seed=args.data_seed)

    # prepare classifier model
    classifier = tx.modules...
    # prepare augmentation model
    augmentation_model = tx.modules...
    # prepare generator
    generator = MetaAugmentationWrapper(augmentation_model)
    # prepare joint trainer
    trainer = DaRlTrainer(classifier, generator, args)

    # pre-train
    trainer.pretrain(examples['train'])

    # joint train
    for epoch in range(args.epochs):
        trainer.train_epoch(
            epoch=epoch,
            train_examples=examples['train'])

        dev_acc = classifier.evaluate(examples['dev'])


if __name__ == '__main__':
    main()
