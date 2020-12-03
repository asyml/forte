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
"""Example of building a reinforcement learning based,
data augmentation enhanced
sentence classifier based on pre-trained BERT model,
with IMDB Dataset.
"""

import argparse
import functools
import logging
import os

import torch
import torch.nn.functional as F
import texar.torch as tx
from pytorch_pretrained_bert.modeling import BertForMaskedLM

from examples.da_rl import config_data
from examples.da_rl import config_classifier
from examples.da_rl.utils import data_utils, model_utils
from forte.models.da_rl.generator_trainer import MetaAugmentationWrapper


parser = argparse.ArgumentParser()
parser.add_argument(
    '--pretrained-model-name', type=str, default='bert-base-uncased',
    choices=tx.modules.BERTEncoder.available_checkpoints(),
    help="Name of the pre-trained downstream checkpoint to load.")
parser.add_argument(
    "--output-dir", default="output/",
    help="The output directory where the model checkpoints will be written.")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="Path to a model checkpoint (including bert modules) to restore from.")
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
    '--classifier-pretrain-epoch', type=int, default=3,
    help="number of epochs to pretrain the classifier")
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.root.setLevel(logging.INFO)
config_downstream = {
    k: v for k, v in config_classifier.__dict__.items()
    if not k.startswith('__') and k != "hyperparams"}


def run():
    """
    Builds the model and runs.
    """
    tx.utils.maybe_create_dir(args.output_dir)

    # Loads data
    num_train_data = config_data.num_train_data

    # Builds data augmentation BERT
    aug_model = BertForMaskedLM.from_pretrained(args.augmentation_model_name)
    aug_model.to(device)

    aug_tokenizer = tx.data.BERTTokenizer(
        pretrained_model_name=args.augmentation_model_name)

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
        optimizer_grouped_parameters, betas=(0.9, 0.999), eps=1e-6, lr=aug_lr)

    aug_wrapper = MetaAugmentationWrapper(aug_model, aug_optim, aug_tokenizer,
                                          device, args.num_aug)

    # Builds downstream BERT
    model = tx.modules.BERTClassifier(
        pretrained_model_name=args.pretrained_model_name,
        hparams=config_downstream)
    model.to(device)

    num_train_steps = int(num_train_data / config_data.train_batch_size
                          * config_data.max_train_epoch)
    num_warmup_steps = int(num_train_steps
                           * config_data.warmup_proportion)

    # Builds learning rate decay scheduler
    classifier_lr = 4e-5

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
        opt_params, betas=(0.9, 0.999), eps=1e-6, lr=classifier_lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, functools.partial(model_utils.get_lr_multiplier,
                                 total_steps=num_train_steps,
                                 warmup_steps=num_warmup_steps))

    train_dataset = tx.data.RecordData(
        hparams=config_data.train_hparam, device=device)
    eval_dataset = tx.data.RecordData(
        hparams=config_data.eval_hparam, device=device)
    test_dataset = tx.data.RecordData(
        hparams=config_data.test_hparam, device=device)

    iterator = tx.data.DataIterator(
        {"train": train_dataset, "eval": eval_dataset, "test": test_dataset}
    )

    eval_data_iterator = tx.data.DataIterator({"eval": eval_dataset})
    eval_data_iterator.switch_to_dataset("eval")

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

    def _pre_train_classifier_epoch():
        iterator.switch_to_dataset("train")
        model.train()

        for _ in range(args.classifier_pretrain_epoch):
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

                dis_steps = config_data.display_steps
                if dis_steps > 0 and step % dis_steps == 0:
                    logging.info("step: %d; loss: %f", step, loss)

    def _train_epoch():
        r"""Trains on the training set, and evaluates on the dev set
        periodically.
        """
        iterator.switch_to_dataset("train")
        model.train()
        optim.zero_grad()

        for batch in iterator:
            input_ids = batch["input_ids"]
            input_mask = batch["input_mask"]
            segment_ids = batch["segment_ids"]
            labels = batch["label_ids"]

            # train augmentation model params
            aug_wrapper.reset_model()

            num_examples = len(input_ids)
            for i in range(num_examples):
                features = (input_ids[i], input_mask[i],
                            segment_ids[i], labels[i])

                # augmented example with params phi exposed
                aug_probs, input_mask_aug, segment_ids_aug, label_ids_aug = \
                    aug_wrapper.augment_example(features)

                input_length_aug = ((input_mask_aug == 1).int()).sum(dim=1)

                model.zero_grad()
                logits, _ = model(aug_probs, input_length_aug, segment_ids_aug)
                loss = _compute_loss(logits, label_ids_aug)

                meta_model = aug_wrapper.update_meta_classifier(loss,
                                                                model,
                                                                optim)

                # compute grads of aug model on eval data
                for val_batch in eval_data_iterator:
                    val_input_ids = val_batch["input_ids"]
                    val_segment_ids = val_batch["segment_ids"]
                    val_labels = val_batch["label_ids"]
                    val_input_length = \
                        (1 - (val_input_ids == 0).int()).sum(dim=1)
                    val_logits, _ = meta_model(val_input_ids,
                                               val_input_length,
                                               val_segment_ids)
                    val_loss = _compute_loss(val_logits, val_labels)
                    # L_{val}(theta'(phi))
                    # apply gradients to phi
                    val_loss = val_loss / num_examples / args.num_aug \
                               / len(eval_data_iterator)
                    val_loss.backward()

            aug_wrapper.update_phi()

            # train classifier with augmented batch
            input_probs, input_masks, segment_ids, label_ids = \
                aug_wrapper.augment_batch(input_ids, input_mask,
                                          segment_ids, labels)

            input_length = ((input_masks == 1).int()).sum(dim=1)

            optim.zero_grad()
            logits, _ = model(input_probs, input_length, segment_ids)
            loss = _compute_loss(logits, label_ids)
            loss.backward()
            optim.step()
            scheduler.step()
            step = scheduler.last_epoch

            dis_steps = config_data.display_steps
            if dis_steps > 0 and step % dis_steps == 0:
                logging.info("step: %d; loss: %f", step, loss)

            eval_steps = config_data.eval_steps
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
    def _test_epoch(test_file):
        """Does predictions on the test set.
        """
        iterator.switch_to_dataset("test")
        model.eval()

        _all_preds = []
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

            _all_preds.extend(preds.tolist())

        logging.info("test accu: %.4f; loss: %.4f; nsamples: %d",
                     avg_rec.avg(0), avg_rec.avg(1), nsamples)

        output_file = os.path.join(args.output_dir, test_file)
        with open(output_file, "w+") as writer:
            writer.write("\n".join(str(p) for p in _all_preds))
        logging.info("test output written to %s", output_file)

    _pre_train_classifier_epoch()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    if args.do_train:
        for k in range(config_data.max_train_epoch):
            logging.info("training epoch %d", k)
            _train_epoch()

        states = {
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        torch.save(states, os.path.join(args.output_dir, 'model.ckpt'))
    if args.do_eval:
        _eval_epoch()
    if args.do_test:
        _test_epoch("test_results.tsv")


def prepare_data(csv_data_dir):
    """Prepares data.
    """
    logging.info("Loading data")

    if config_data.pickle_data_dir is None:
        output_dir = csv_data_dir
    else:
        output_dir = config_data.pickle_data_dir
    tx.utils.maybe_create_dir(output_dir)

    processor = data_utils.IMDbProcessor()

    num_classes = len(processor.get_labels())
    num_train_data = len(processor.get_train_examples(csv_data_dir))
    logging.info(
        'num_classes:%d; num_train_data:%d', num_classes, num_train_data)

    tokenizer = tx.data.BERTTokenizer(
        pretrained_model_name=args.pretrained_model_name)

    data_utils.prepare_record_data(
        processor=processor,
        tokenizer=tokenizer,
        data_dir=csv_data_dir,
        max_seq_length=config_data.max_seq_length,
        output_dir=output_dir,
        feature_types=config_data.feature_types)


def main():
    if not os.path.isfile("data/IMDB/train.pkl") \
            or not os.path.isfile("data/IMDB/eval.pkl") \
            or not os.path.isfile("data/IMDB/predict.pkl"):
        prepare_data("data/IMDB")

    run()


if __name__ == "__main__":
    main()
