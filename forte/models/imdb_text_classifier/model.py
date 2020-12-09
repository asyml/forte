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
from forte.processors.data_augment.algorithms.UDA import UDAIterator


class IMDBClassifier:
    """
    A baseline text classifier for the IMDB dataset.
    The input data should be CSV format with columns (content label id).
    An example usage can be found at examples/text_classification.
    """

    def __init__(self, config_data, config_classifier, checkpoint=None,
        pretrained_model_name="bert-base-uncased"):
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

        # num_classes = len(processor.get_labels())
        # num_train_data = len(processor.get_train_examples(csv_data_dir))
        # logging.info(
        #     'num_classes:%d; num_train_data:%d', num_classes, num_train_data)

        tokenizer = tx.data.BERTTokenizer(
            pretrained_model_name=self.pretrained_model_name)

        data_utils.prepare_record_data(
            processor=processor,
            tokenizer=tokenizer,
            data_dir=csv_data_dir,
            max_seq_length=self.config_data.max_seq_length,
            output_dir=output_dir,
            feature_types=self.config_data.feature_types,
            unsup_feature_types=self.config_data.unsup_feature_types,
            sup_size_limit=self.config_data.num_train_data,
            unsup_bt_file=self.config_data.unsup_bt_file,
        )

    def run(self, do_train, do_eval, do_test, output_dir="output/"):
        """
        Builds the model and runs.
        """
        tx.utils.maybe_create_dir(output_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.root.setLevel(logging.INFO)

        # Loads data
        num_train_data = self.config_data.num_train_data

        hparams = {
            k: v for k, v in self.config_classifier.__dict__.items()
            if not k.startswith('__') and k != "hyperparams"}

        # Builds BERT
        model = tx.modules.BERTClassifier(
            pretrained_model_name=self.pretrained_model_name,
            hparams=hparams)
        model.to(device)

        num_train_steps = int(num_train_data / self.config_data.train_batch_size
                            * self.config_data.max_train_epoch)
        num_warmup_steps = int(num_train_steps
                            * self.config_data.warmup_proportion)

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

        train_dataset = tx.data.RecordData(
            hparams=self.config_data.train_hparam, device=device)
        eval_dataset = tx.data.RecordData(
            hparams=self.config_data.eval_hparam, device=device)
        test_dataset = tx.data.RecordData(
            hparams=self.config_data.test_hparam, device=device)

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

            output_file = os.path.join(output_dir, "test_results.tsv")
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
                if self.config_data.eval_steps == -1:
                    _eval_epoch()
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

    def run_uda(self, do_train, do_eval, do_test, output_dir="output/"):
        """
        Builds the model and runs.
        """
        tx.utils.maybe_create_dir(output_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.root.setLevel(logging.INFO)

        # Loads data
        num_train_data = self.config_data.num_train_data

        hparams = {
            k: v for k, v in self.config_classifier.__dict__.items()
            if not k.startswith('__') and k != "hyperparams"}

        # Builds BERT
        model = tx.modules.BERTClassifier(
            pretrained_model_name=self.pretrained_model_name,
            hparams=hparams)
        model.to(device)

        num_train_steps = int(num_train_data / self.config_data.train_batch_size
                            * self.config_data.max_train_epoch)
        num_warmup_steps = int(num_train_steps
                            * self.config_data.warmup_proportion)

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

        train_dataset = tx.data.RecordData(
            hparams=self.config_data.train_hparam, device=device)
        eval_dataset = tx.data.RecordData(
            hparams=self.config_data.eval_hparam, device=device)
        test_dataset = tx.data.RecordData(
            hparams=self.config_data.test_hparam, device=device)
        unsup_dataset = tx.data.RecordData(
            hparams=self.config_data.unsup_hparam, device=device)

        iterator = tx.data.DataIterator(
            {"train": train_dataset, "eval": eval_dataset, "test": test_dataset}
        )

        unsup_iterator = tx.data.DataIterator(
            {"unsup": unsup_dataset}
        )

        def unsup_forward_fn(batch):
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]
            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            aug_input_ids = batch["aug_input_ids"]
            aug_segment_ids = batch["aug_segment_ids"]
            aug_input_length = (1 - (aug_input_ids == 0).int()).sum(dim=1)

            logits, _ = model(input_ids, input_length, segment_ids)
            logits = logits.detach()  # gradient does not propagate back to original input
            aug_logits, _ = model(aug_input_ids, aug_input_length, aug_segment_ids)
            return logits, aug_logits

        uda_iterator = UDAIterator(
            iterator,
            unsup_iterator,
            softmax_temperature=1.0,
            confidence_threshold=-1,
            reduction="mean")

        uda_iterator.switch_to_dataset_unsup("unsup")
        uda_iterator.switch_to_dataset("train", use_unsup=True)
        uda_iterator = iter(uda_iterator) # call iter() to initialize the internal iterators

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

        def _compute_loss_tsa(logits, labels, global_step, num_train_steps):
            r"""Compute loss.
            """
            loss = 0
            log_probs = F.log_softmax(logits)
            one_hot_labels = torch.zeros_like(log_probs, dtype=torch.float).to(device)
            one_hot_labels.scatter_(1, labels.view(-1, 1), 1)
            tgt_label_prob = one_hot_labels.clone()

            per_example_loss = -(one_hot_labels * log_probs).sum(dim=-1)
            loss_mask = torch.ones_like(per_example_loss, dtype=per_example_loss.dtype).to(device)
            correct_label_probs = (one_hot_labels * torch.exp(log_probs)).sum(dim=-1)

            if self.config_data.tsa:
                tsa_start = 1. / model.num_classes
                tsa_threshold = model_utils.get_tsa_threshold(
                    self.config_data.tsa_schedule, global_step,
                    num_train_steps, start=tsa_start, end=1)
                larger_than_threshold = torch.gt(correct_label_probs, tsa_threshold)
                loss_mask = loss_mask * (1 - larger_than_threshold.float())
            else:
                tsa_threshold = 1
            
            loss_mask = loss_mask.detach()
            per_example_loss = per_example_loss * loss_mask
            loss_mask_sum = loss_mask.sum()
            loss = per_example_loss.sum()
            if loss_mask_sum > 0:
                loss = loss / loss_mask_sum
            return loss

        def _train_epoch():
            r"""Trains on the training set, and evaluates on the dev set
            periodically.
            """
            model.train()
            uda_iterator.switch_to_dataset("train", use_unsup=True)
            iter(uda_iterator)
            nsamples = 0
            for batch, unsup_batch in uda_iterator:
                optim.zero_grad()
                input_ids = batch["input_ids"]
                segment_ids = batch["segment_ids"]
                labels = batch["label_ids"]

                batch_size = input_ids.size()[0]
                nsamples += batch_size

                input_length = (1 - (input_ids == 0).int()).sum(dim=1)
                
                # sup loss
                logits, _ = model(input_ids, input_length, segment_ids)
                loss = _compute_loss_tsa(logits, labels, scheduler.last_epoch,\
                    num_train_steps)
                # unsup loss
                unsup_logits, unsup_aug_logits = unsup_forward_fn(unsup_batch)
                unsup_loss = uda_iterator.calculate_uda_loss(unsup_logits, unsup_aug_logits)

                loss = loss + unsup_loss # unsup coefficient = 1
                loss.backward()
                optim.step()
                scheduler.step()
                step = scheduler.last_epoch

                dis_steps = self.config_data.display_steps
                if dis_steps > 0 and step % dis_steps == 0:
                    logging.info("step: %d; loss: %f, unsup_loss %f", step, loss, unsup_loss)

                eval_steps = self.config_data.eval_steps
                if eval_steps > 0 and step % eval_steps == 0:
                    _eval_epoch()
                    model.train()
                    # uda_iterator.switch_to_dataset("train", use_unsup=True)
            print("Train nsamples:", nsamples)

        @torch.no_grad()
        def _eval_epoch():
            """Evaluates on the dev set.
            """
            uda_iterator.switch_to_dataset("eval", use_unsup=False)
            model.eval()

            nsamples = 0
            avg_rec = tx.utils.AverageRecorder()
            for batch, _ in uda_iterator:
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
            uda_iterator.switch_to_dataset("test", use_unsup=False)
            model.eval()

            _all_preds = []
            for batch, _ in uda_iterator:
                input_ids = batch["input_ids"]
                segment_ids = batch["segment_ids"]

                input_length = (1 - (input_ids == 0).int()).sum(dim=1)

                _, preds = model(input_ids, input_length, segment_ids)

                _all_preds.extend(preds.tolist())

            output_file = os.path.join(output_dir, "test_results.tsv")
            with open(output_file, "w+") as writer:
                writer.write("\n".join(str(p) for p in _all_preds))
            logging.info("test output written to %s", output_file)

        if self.checkpoint:
            ckpt = torch.load(self.checkpoint)
            model.load_state_dict(ckpt['model'])
            optim.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
        if do_train:
            for i in range(self.config_data.max_train_epoch):
                print("Epoch", i)
                _train_epoch()
                if self.config_data.eval_steps == -1:
                    _eval_epoch() # eval after epoch because switch_dataset just resets the iterator
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
