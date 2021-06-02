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
"""
Example of building a reinforcement learning based,
data augmentation enhanced sentence classifier
based on pre-trained BERT model.
"""
import argparse
import functools
import logging
import os

import torch
import torch.nn.functional as F
import texar.torch as tx
from transformers import BertForMaskedLM

from config import config_data, config_classifier
from utils import model_utils
from forte.models.da_rl import MetaAugmentationWrapper, TexarBertMetaModule

parser = argparse.ArgumentParser()
parser.add_argument(
    '--pretrained-model-name', type=str, default='bert-base-uncased',
    choices=tx.modules.BERTEncoder.available_checkpoints(),
    help="Name of the pre-trained downstream checkpoint to load.")
parser.add_argument(
    "--output-dir", default="output/",
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    "--do-train", action="store_true", help="Whether to run training.")
parser.add_argument(
    "--do-eval", action="store_true",
    help="Whether to run eval on the dev set.")
parser.add_argument(
    "--do-test", action="store_true",
    help="Whether to run test on the test set.")
parser.add_argument(
    '--augmentation-model-name', type=str, default='bert-base-uncased',
    choices=tx.modules.BERTEncoder.available_checkpoints(),
    help="Name of the pre-trained augmentation model checkpoint to load.")
parser.add_argument(
    '--num-aug', type=int, default=4,
    help="number of augmentation samples when fine-tuning aug model")
parser.add_argument(
    '--classifier-pretrain-epoch', type=int, default=10,
    help="number of epochs to pretrain the classifier")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.root.setLevel(logging.INFO)


class RLAugmentClassifierTrainer:
    def __init__(self):
        self._prepare_data_iterator()
        self._init_aug_model()
        self._init_classifier()

    def _prepare_data_iterator(self):
        tx.utils.maybe_create_dir(args.output_dir)

        # Loads data
        num_train_data = config_data.num_train_data
        self.num_train_steps = \
            int(num_train_data / config_data.train_batch_size *
                config_data.max_train_epoch)

        train_dataset = tx.data.RecordData(
            hparams=config_data.train_hparam, device=device)
        val_dataset = tx.data.RecordData(
            hparams=config_data.eval_hparam, device=device)
        test_dataset = tx.data.RecordData(
            hparams=config_data.test_hparam, device=device)
        self.iterator = tx.data.DataIterator(
            {"train": train_dataset,
             "dev": val_dataset,
             "test": test_dataset}
        )

        self.val_data_iterator = tx.data.DataIterator({"dev": val_dataset})
        self.val_data_iterator.switch_to_dataset("dev")

    def _init_aug_model(self):
        # pylint: disable=protected-access
        # Builds data augmentation BERT
        aug_model = BertForMaskedLM.from_pretrained(
            args.augmentation_model_name)
        aug_model.to(device)
        aug_tokenizer = tx.data.BERTTokenizer(
            pretrained_model_name=args.augmentation_model_name)
        input_mask_ids = aug_tokenizer._map_token_to_id('[MASK]')
        # Builds augmentation optimizer
        aug_lr = 4e-5
        param_optimizer = list(aug_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if
                        any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}]
        aug_optim = tx.core.BertAdam(
            optimizer_grouped_parameters, betas=(0.9, 0.999),
            eps=1e-6, lr=aug_lr)
        # Builds data augmentation wrapper
        self.aug_wrapper = MetaAugmentationWrapper(
            aug_model, aug_optim, input_mask_ids, device, args.num_aug)

    def _init_classifier(self):
        # Builds BERT for classification task.
        config_downstream = {
            k: v for k, v in config_classifier.__dict__.items()
            if not k.startswith('__') and k != "hyperparams"}

        self.classifier = tx.modules.BERTClassifier(
            pretrained_model_name=args.pretrained_model_name,
            hparams=config_downstream)
        self.classifier.to(device)

        # Builds learning rate decay scheduler
        classifier_lr = 4e-5
        vars_with_decay = []
        vars_without_decay = []
        for name, param in self.classifier.named_parameters():
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
        self.optim = tx.core.BertAdam(
            opt_params, betas=(0.9, 0.999), eps=1e-6, lr=classifier_lr)
        num_warmup_steps = int(self.num_train_steps
                               * config_data.warmup_proportion)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optim, functools.partial(model_utils.get_lr_multiplier,
                                          total_steps=self.num_train_steps,
                                          warmup_steps=num_warmup_steps))

    def pre_train_classifier_epoch(self):
        r"""Pre-trains model on the training set
        for better weight initialization.
        """
        self.iterator.switch_to_dataset("train")
        self.classifier.train()

        for _ in range(args.classifier_pretrain_epoch):
            for batch in self.iterator:
                self.optim.zero_grad()

                input_ids = batch["input_ids"]
                segment_ids = batch["segment_ids"]
                labels = batch["label_ids"]
                input_length = (1 - (input_ids == 0).int()).sum(dim=1)

                logits, _ = self.classifier(
                    input_ids, input_length, segment_ids)
                loss = self._compute_loss(logits, labels)

                loss.backward()
                self.optim.step()
                self.scheduler.step()

    def train_epoch(self):
        r"""Trains on the training set, and evaluates on the validation set
        periodically.
        """
        self.iterator.switch_to_dataset("train")
        self.classifier.train()
        self.optim.zero_grad()

        for batch in self.iterator:
            input_ids = batch["input_ids"]
            input_mask = batch["input_mask"]
            segment_ids = batch["segment_ids"]
            labels = batch["label_ids"]

            # Train augmentation model params phi.
            self.aug_wrapper.reset_model()
            # Iterate over training instances.
            num_instances = len(input_ids)
            for i in range(num_instances):
                features = (input_ids[i], input_mask[i],
                            segment_ids[i], labels[i])

                # Augmented instance with params phi exposed
                aug_probs, input_mask_aug, segment_ids_aug, label_ids_aug = \
                    self.aug_wrapper.augment_instance(features)

                # Compute classifier loss.
                self.classifier.zero_grad()
                input_length_aug = ((input_mask_aug == 1).int()).sum(dim=1)
                logits, _ = self.classifier(
                    aug_probs, input_length_aug, segment_ids_aug)
                loss = self._compute_loss(logits, label_ids_aug)
                # Update classifier params on meta_model.
                meta_model = TexarBertMetaModule(self.classifier)
                meta_model = self.aug_wrapper.update_meta_model(
                    meta_model, loss, self.classifier, self.optim)

                # Compute grads of aug_model on validation data.
                for val_batch in self.val_data_iterator:  # one batch
                    val_input_ids = val_batch["input_ids"]
                    val_segment_ids = val_batch["segment_ids"]
                    val_labels = val_batch["label_ids"]
                    val_input_length = \
                        (1 - (val_input_ids == 0).int()).sum(dim=1)
                    val_logits, _ = meta_model(val_input_ids,
                                               val_input_length,
                                               val_segment_ids)
                    val_loss = self._compute_loss(val_logits, val_labels)
                    val_loss = val_loss / num_instances / args.num_aug \
                               / len(self.val_data_iterator)
                    val_loss.backward()

            # Update aug_model param phi.
            self.aug_wrapper.update_phi()

            # Train classifier with augmented batch
            input_probs, input_masks, segment_ids, label_ids = \
                self.aug_wrapper.augment_batch((input_ids, input_mask,
                                                segment_ids, labels))

            input_length = ((input_masks == 1).int()).sum(dim=1)
            self.optim.zero_grad()
            logits, _ = self.classifier(input_probs, input_length, segment_ids)
            loss = self._compute_loss(logits, label_ids)
            loss.backward()
            self.optim.step()
            self.scheduler.step()
            self._display_logging(loss)

    @torch.no_grad()
    def eval_epoch(self):
        """Evaluates on the dev set.
        """
        self.iterator.switch_to_dataset("dev")
        self.classifier.eval()

        nsamples = 0
        avg_rec = tx.utils.AverageRecorder()
        for batch in self.iterator:
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]
            labels = batch["label_ids"]
            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            logits, preds = \
                self.classifier(input_ids, input_length, segment_ids)
            loss = self._compute_loss(logits, labels)
            accu = tx.evals.accuracy(labels, preds)

            batch_size = input_ids.size()[0]
            avg_rec.add([accu, loss], batch_size)
            nsamples += batch_size
        logging.info("eval accu: %.4f; loss: %.4f; nsamples: %d",
                     avg_rec.avg(0), avg_rec.avg(1), nsamples)

    @torch.no_grad()
    def test_epoch(self, test_file):
        """Does predictions on the test set.
        """
        self.iterator.switch_to_dataset("test")
        self.classifier.eval()

        _all_preds = []
        nsamples = 0
        avg_rec = tx.utils.AverageRecorder()
        for batch in self.iterator:
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]
            labels = batch["label_ids"]
            input_length = (1 - (input_ids == 0).int()).sum(dim=1)

            logits, preds = \
                self.classifier(input_ids, input_length, segment_ids)
            loss = self._compute_loss(logits, labels)
            accu = tx.evals.accuracy(labels, preds)

            batch_size = input_ids.size()[0]
            avg_rec.add([accu, loss], batch_size)
            nsamples += batch_size

            _all_preds.extend(preds.tolist())

        logging.info("test accu: %.4f; loss: %.4f; nsamples: %d",
                     avg_rec.avg(0), avg_rec.avg(1), nsamples)

        output_file = os.path.join(args.output_dir, test_file)
        with open(output_file, "w+") as writer:
            writer.write("\n".join(str(p) for p in _all_preds))
        logging.info("test output written to %s", output_file)

    def _compute_loss(self, logits, labels):
        r"""Compute loss.
        """
        if self.classifier.is_binary:
            loss = F.binary_cross_entropy(
                logits.view(-1), labels.view(-1), reduction='mean')
        else:
            loss = F.cross_entropy(
                logits.view(-1, self.classifier.num_classes),
                labels.view(-1), reduction='mean')
        return loss

    def _display_logging(self, loss):
        step = self.scheduler.last_epoch
        dis_steps = config_data.display_steps
        if dis_steps > 0 and step % dis_steps == 0:
            logging.info("step: %d; loss: %f", step, loss)

        eval_steps = config_data.eval_steps
        if eval_steps > 0 and step % eval_steps == 0:
            self._eval_epoch()
            self.classifier.train()


def main():
    trainer = RLAugmentClassifierTrainer()
    trainer.pre_train_classifier_epoch()
    if args.do_train:
        for k in range(config_data.max_train_epoch):
            logging.info("training epoch %d", k)
            trainer.train_epoch()
    if args.do_eval:
        trainer.eval_epoch()
    if args.do_test:
        trainer.test_epoch("test_results.tsv")


if __name__ == "__main__":
    main()
