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

import functools
import logging
import os

import torch
import torch.nn.functional as F
import texar.torch as tx

# pylint: disable=no-name-in-module
from forte.models.imdb_text_classifier.utils import data_utils, model_utils


class IMDBClassifier:
    """
    A baseline text classifier for the IMDB dataset.
    The input data should be CSV format with columns (content label id).
    An example usage can be found at examples/text_classification.
    """

    def __init__(self, config_data, config_classifier, checkpoint=None, pretrained_model_name="bert-base-uncased"):
        """Constructs the text classifier.
        Args:
            config_data: string, data config file.
        """
        self.config_data = config_data
        self.config_classifier = config_classifier
        self.checkpoint = checkpoint
        self.pretrained_model_name = pretrained_model_name
    
    def prepare_data(self, csv_data_dir):
        """Prepares data.
        """
        logging.info("Loading data")

        if self.config_data.pickle_data_dir is None:
            output_dir = csv_data_dir
        else:
            output_dir = self.config_data.pickle_data_dir
        tx.utils.maybe_create_dir(output_dir)
        
        processor = data_utils.IMDbProcessor()

        num_classes = len(processor.get_labels())
        num_train_data = len(processor.get_train_examples(csv_data_dir))
        logging.info(
            'num_classes:%d; num_train_data:%d' % (num_classes, num_train_data))

        tokenizer = tx.data.BERTTokenizer(
            pretrained_model_name=self.pretrained_model_name)

        data_utils.prepare_record_data(
            processor=processor,
            tokenizer=tokenizer,
            data_dir=csv_data_dir,
            max_seq_length=self.config_data.max_seq_length,
            output_dir=output_dir,
            feature_types=self.config_data.feature_types)

    def run(self, do_train, do_eval, do_test, output_dir="output/"):
        """
        Builds the model and runs.
        """
        tx.utils.maybe_create_dir(output_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.root.setLevel(logging.INFO)

        # Loads data
        num_train_data = self.config_data.num_train_data

        # config_downstream = importlib.import_module(args.config_downstream)
        hparams = {
            k: v for k, v in self.config_classifier.__dict__.items()
            if not k.startswith('__') and k != "hyperparams"}

        # Builds BERT
        model = tx.modules.BERTClassifier(
            pretrained_model_name=self.pretrained_model_name,
            hparams=hparams)
        model.to(device)

        num_train_steps = int(num_train_data / self.config_data.train_batch_size *
                              self.config_data.max_train_epoch)
        num_warmup_steps = int(num_train_steps * self.config_data.warmup_proportion)

        # Builds learning rate decay scheduler
        static_lr = 2e-5

        vars_with_decay = []
        vars_without_decay = []
        for name, param in model.named_parameters():
            if 'layer_norm' in name or name.endswith('bias'):
                vars_without_decay.append(param)
            else:
                vars_with_decay.append(param)

        opt_params = [{
            'params': vars_with_decay,
            'weight_decay': 0.01,
        }, {
            'params': vars_without_decay,
            'weight_decay': 0.0,
        }]
        optim = tx.core.BertAdam(
            opt_params, betas=(0.9, 0.999), eps=1e-6, lr=static_lr)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, functools.partial(model_utils.get_lr_multiplier,
                                     total_steps=num_train_steps,
                                     warmup_steps=num_warmup_steps))

        train_dataset = tx.data.RecordData(hparams=self.config_data.train_hparam,
                                           device=device)
        eval_dataset = tx.data.RecordData(hparams=self.config_data.eval_hparam,
                                          device=device)
        test_dataset = tx.data.RecordData(hparams=self.config_data.test_hparam,
                                          device=device)

        iterator = tx.data.DataIterator(
            {"train": train_dataset, "eval": eval_dataset, "test": test_dataset}
        )

        def _compute_loss(logits, labels):
            r"""Compute loss.
            """
            if model.is_binary:
                loss = F.binary_cross_entropy(
                    logits.view(-1), labels.view(-1), reduction='mean')
            else:
                loss = F.cross_entropy(
                    logits.view(-1, model.num_classes),
                    labels.view(-1), reduction='mean')
            return loss

        def _train_epoch():
            r"""Trains on the training set, and evaluates on the dev set
            periodically.
            """
            iterator.switch_to_dataset("train")
            model.train()

            for batch in iterator:
                optim.zero_grad()
                input_ids = batch["input_ids"]
                segment_ids = batch["segment_ids"]
                labels = batch["label_ids"]

                input_length = (1 - (input_ids == 0).int()).sum(dim=1)

                logits, _ = model(input_ids, input_length, segment_ids)

                loss = _compute_loss(logits, labels)
                loss.backward()
                optim.step()
                scheduler.step()
                step = scheduler.last_epoch

                dis_steps = self.config_data.display_steps
                if dis_steps > 0 and step % dis_steps == 0:
                    logging.info("step: %d; loss: %f", step, loss)

                eval_steps = self.config_data.eval_steps
                if eval_steps > 0 and step % eval_steps == 0:
                    _eval_epoch()
                    model.train()

        @torch.no_grad()
        def _eval_epoch():
            """Evaluates on the dev set.
            """
            iterator.switch_to_dataset("eval")
            model.eval()

            nsamples = 0
            avg_rec = tx.utils.AverageRecorder()
            for batch in iterator:
                input_ids = batch["input_ids"]
                segment_ids = batch["segment_ids"]
                labels = batch["label_ids"]

                input_length = (1 - (input_ids == 0).int()).sum(dim=1)

                logits, preds = model(input_ids, input_length, segment_ids)

                loss = _compute_loss(logits, labels)
                accu = tx.evals.accuracy(labels, preds)
                batch_size = input_ids.size()[0]
                avg_rec.add([accu, loss], batch_size)
                nsamples += batch_size
            logging.info("eval accu: %.4f; loss: %.4f; nsamples: %d",
                        avg_rec.avg(0), avg_rec.avg(1), nsamples)

        @torch.no_grad()
        def _test_epoch():
            """Does predictions on the test set.
            """
            iterator.switch_to_dataset("test")
            model.eval()

            _all_preds = []
            for batch in iterator:
                input_ids = batch["input_ids"]
                segment_ids = batch["segment_ids"]

                input_length = (1 - (input_ids == 0).int()).sum(dim=1)

                _, preds = model(input_ids, input_length, segment_ids)

                _all_preds.extend(preds.tolist())

            output_file = os.path.join(args.output_dir, "test_results.tsv")
            with open(output_file, "w+") as writer:
                writer.write("\n".join(str(p) for p in _all_preds))
            logging.info("test output written to %s", output_file)

        if self.checkpoint:
            ckpt = torch.load(self.checkpoint)
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
        if do_train:
            for _ in range(self.config_data.max_train_epoch):
                _train_epoch()
            states = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
            }
            torch.save(states, os.path.join(output_dir, 'model.ckpt'))

        if do_eval:
            _eval_epoch()

        if do_test:
            _test_epoch()
