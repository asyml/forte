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
A wrapper adding data augmentation to a model with arbitrary tasks.
"""

import random
import texar.torch as tx
from torch.nn import functional as F
import torch

from forte.models.da_rl.magic_model import MetaModule


__all__ = [
    "MetaAugmentationWrapper"
]


class MetaAugmentationWrapper:
    # pylint: disable=line-too-long
    r"""A wrapper adding data augmentation to a model with arbitrary tasks.
    See: https://arxiv.org/pdf/1910.12795.pdf
    There is an example code for this class here:
    https://github.com/tanyuqian/learning-data-manipulation/blob/master/augmentation/generator.py
    Let theta be the parameters of the downstream (classifier) model.
    Let phi be the parameters of the augmentation model.
    Equations to update phi:
    theta'(phi) = theta - \nabla_{theta} L_{train}(theta, phi)
    phi = phi - \nabla_{phi} L_{val}(theta'(phi))
    """

    def __init__(self, augmentation_model, augmentation_optimizer,
                 aug_tokenizer, device, num_aug):
        r"""
        :param augmentation_model:
        A Bert-based language model for data augmentation.
        :param augmentation_optimizer:
        An optimizer.
        :param aug_tokenizer:
        A Bert-based tokenizer.
        :param device:
        The CUDA device to run the model on.
        :param num_aug:
        The number of samples from the LM for an augmented training example.
        See :meth:`_augment_example` for implementation details.
        """
        self._aug_model = augmentation_model
        self._aug_optimizer = augmentation_optimizer
        self._aug_tokenizer = aug_tokenizer
        self._device = device
        self._num_aug = num_aug

    def reset_model(self):
        self._aug_model.train()
        self._aug_model.zero_grad()

    def _augment_example(self, features, num_aug):
        # pylint: disable=protected-access

        init_ids, input_mask, segment_ids, _ = \
            (t.view(1, -1).to(self._device) for t in features)

        len = int(torch.sum(input_mask).item())
        if len >= 4:
            mask_idx = sorted(
                random.sample(list(range(1, len - 1)), max(len // 7, 2)))
        else:
            mask_idx = [1]

        # init_ids[0][mask_idx] = \
        #     self._aug_tokenizer.convert_tokens_to_ids(['[MASK]'])[0]

        input_mask_ids = self._aug_tokenizer._map_token_to_id('[MASK]')
        init_ids[0][mask_idx] = input_mask_ids

        logits = self._aug_model(init_ids, segment_ids, input_mask)[0]

        # Get samples
        aug_probs_all = []
        for _ in range(num_aug):
            # Need a gumbel trick here in order to keep phi as variables.
            # Enable efficient gradient propagation through theta' to phi.
            probs = F.gumbel_softmax(logits, hard=False)
            aug_probs = torch.zeros_like(probs).scatter_(
                1, init_ids[0].unsqueeze(1), 1.)

            for t in mask_idx:
                aug_probs = tx.utils.pad_and_concat(
                    [aug_probs[:t], probs[t:t + 1], aug_probs[t + 1:]], axis=0)

            aug_probs_all.append(aug_probs)

        aug_probs = tx.utils.pad_and_concat(
            [ap.unsqueeze(0) for ap in aug_probs_all], axis=0)

        return aug_probs

    def augment_example(self, features):
        r"""Augment a training example.

        Args:
            features: A tuple of Bert features of one training example.
                (input_ids, input_mask, segment_ids, label_ids).
                `input_ids` is a tensor of Bert token ids.
                It has shape `[seq_len, 1]`.

        Returns:
            A tuple of Bert features of augmented training examples.
                (input_probs_aug, input_mask_aug, segment_ids_aug, label_ids_aug).
                `input_probs_aug` is a tensor of soft Bert embeddings,
                distributions over vocabulary.
                It has shape `[num_aug, seq_len, vocab_size]`.
                It keeps phi as variable so that after passing it to the classifier,
                the gradients of theta will also apply to phi.
        """

        aug_probs = self._augment_example(features, self._num_aug)

        _, input_mask, segment_ids, label_ids = \
            (t.to(self._device).unsqueeze(0) for t in features)
        input_mask_aug = tx.utils.pad_and_concat(
            [input_mask] * self._num_aug, axis=0)
        segment_ids_aug = tx.utils.pad_and_concat(
            [segment_ids] * self._num_aug, axis=0)
        label_ids_aug = tx.utils.pad_and_concat(
            [label_ids] * self._num_aug, axis=0)

        return aug_probs, input_mask_aug, segment_ids_aug, label_ids_aug

    def augment_batch(self, input_ids, input_mask, segment_ids, labels):
        r"""Augment a batch of training examples.

        Args:
            features: A tuple of Bert features of a batch training example.
                (input_ids, input_mask, segment_ids, label_ids).
                `input_ids` is a tensor of Bert token ids.
                It has shape `[batch_size, seq_len, 1]`.

        Returns:
            A tuple of Bert features of augmented training examples.
                (input_probs_aug, input_mask_aug, segment_ids_aug, label_ids_aug).
                `input_probs_aug` is a tensor of soft Bert embeddings,
                It has shape `[batch_size, seq_len, vocab_size]`.
        """

        self._aug_model.eval()

        aug_examples = []
        features = []
        num_example = len(input_ids)
        for i in range(num_example):
            feature = (input_ids[i], input_mask[i], segment_ids[i], labels[i])
            features.append(feature)
            with torch.no_grad():
                aug_probs = self._augment_example(feature, num_aug=1)
                aug_examples.append(aug_probs)

        input_ids_or_probs, input_masks, segment_ids, label_ids = \
            [tx.utils.pad_and_concat(
                [t[i].unsqueeze(0) for t in features], axis=0).to(
                self._device) for i in range(4)]

        num_aug = len(aug_examples[0])

        input_ids_or_probs_aug = []
        for i in range(num_aug):
            for aug_probs in aug_examples:
                input_ids_or_probs_aug.append(aug_probs[i:i + 1])
        input_ids_or_probs_aug = tx.utils.pad_and_concat(
            input_ids_or_probs_aug, axis=0).to(self._device)

        inputs_onehot = torch.zeros_like(
            input_ids_or_probs_aug[:len(input_ids_or_probs)]).scatter_(
            2, input_ids_or_probs.unsqueeze(2), 1.)
        input_ids_or_probs = tx.utils.pad_and_concat(
            [inputs_onehot, input_ids_or_probs_aug], axis=0).to(self._device)

        segment_ids = tx.utils.pad_and_concat(
            [segment_ids] * (num_aug + 1), axis=0).to(self._device)
        input_masks = tx.utils.pad_and_concat(
            [input_masks] * (num_aug + 1), axis=0).to(self._device)
        label_ids = tx.utils.pad_and_concat(
            [label_ids] * (num_aug + 1), axis=0).to(self._device)

        return input_ids_or_probs, input_masks, segment_ids, label_ids

    def update_meta_classifier(self, loss, classifier, classifier_optimizer):
        r"""Update parameters theta.

        Args:
            loss: The loss of the downstream classifier that have taken
                the augmented training examples.
            classifier: The downstream classifier.
            classifier_optimizer: The optimizer for the classifier.

        Returns:
            A meta model of :class:`~forte.forte.models.da_rl.MetaModule`.
                Meta model is used to calculate
                \nabla_{phi} L_{val}(theta'(phi)),
                where it needs gradients applied to phi.
                Meta model copies classifier's states to perform parameter
                updates, and later applies gradient change to theta.
        """

        meta_model = MetaModule(classifier)

        # grads_theta(phi) = \nabla_{theta} L_{train}(theta, phi)
        grads_theta = self._calculate_grads(
            loss, classifier, classifier_optimizer)

        # theta'(phi) = theta - grads_theta(phi)
        meta_model.update_params(grads_theta)

        return meta_model

    @staticmethod
    def _calculate_grads(loss, classifier, classifier_optimizer):
        grads = torch.autograd.grad(
            loss, [param for name, param in classifier.named_parameters()],
            create_graph=True)
        grads = {param: grads[i] for i, (name, param) in enumerate(
            classifier.named_parameters())}
        deltas = _adam_delta(classifier_optimizer, classifier, grads)
        return deltas

    def update_phi(self):
        # phi = phi - \nabla_{phi} L_{val}(theta'(phi))
        self._aug_optimizer.step()


def _adam_delta(optimizer, model, grads):
    # pylint: disable=line-too-long
    r"""compute adam delta function for texar-pytorch optimizer with
    tx.core.BertAdam

    See more implementation details from:
    https://github.com/asyml/texar-pytorch/blob/master/texar/torch/core/optimization.py#L398
    """

    deltas = {}
    for group in optimizer.param_groups:
        for param in group['params']:
            grad = grads[param]
            state = optimizer.state[param]

            exp_avg, exp_avg_sq = state['next_m'], state['next_v']

            beta1, beta2 = group['betas']

            if group['weight_decay'] != 0:
                grad = grad + group['weight_decay'] * param.data

            exp_avg = exp_avg * beta1 + (1. - beta1) * grad
            exp_avg_sq = exp_avg_sq * beta2 + (1. - beta2) * grad * grad
            denom = exp_avg_sq.sqrt() + group['eps']

            # no bias correction
            # bias_correction1 = 1. - beta1 ** step
            # bias_correction2 = 1. - beta2 ** step
            # step_size = group['lr'] * math.sqrt(
            #     bias_correction2) / bias_correction1
            step_size = group['lr']

            deltas[param] = -step_size * exp_avg / denom

    param_to_name = {param: name for name, param in model.named_parameters()}

    return {param_to_name[param]: delta for param, delta in deltas.items()}


# def _adam_delta(optimizer, model, grads):
#     r"""compute adam delta function for torch optimizer with
#     from torch import optim"""
#
#     deltas = {}
#     for group in optimizer.param_groups:
#         for param in group['params']:
#             grad = grads[param]
#             state = optimizer.state[param]
#
#             exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#             beta1, beta2 = group['betas']
#
#             step = state['step'] + 1
#
#             if group['weight_decay'] != 0:
#                 grad = grad + group['weight_decay'] * param.data
#
#             exp_avg = exp_avg * beta1 + (1. - beta1) * grad
#             exp_avg_sq = exp_avg_sq * beta2 + (1. - beta2) * grad * grad
#             denom = exp_avg_sq.sqrt() + group['eps']
#
#             bias_correction1 = 1. - beta1 ** step
#             bias_correction2 = 1. - beta2 ** step
#             step_size = group['lr'] * math.sqrt(
#                 bias_correction2) / bias_correction1
#
#             deltas[param] = -step_size * exp_avg / denom
#
#     param_to_name = {param: name for name, param in model.named_parameters()}
#
#     return {param_to_name[param]: delta for param, delta in deltas.items()}
