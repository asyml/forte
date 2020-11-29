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
from torch.nn import functional as F
import torch
import random
import math

from forte.models.da_rl.magic_model import MetaModule


class MetaAugmentationWrapper:
    '''
    a wrapper adding data augmentation to a model with arbitrary tasks
    see: https://arxiv.org/pdf/1910.12795.pdf

    let theta be the parameters of the classifer model
    let phi be the parameters of the augmentation model

    '''
    def __init__(self, augmentation_model, augmentation_optimizer, aug_tokenizer,
                 device, num_aug):
        self._aug_model = augmentation_model
        self._aug_optimizer = augmentation_optimizer
        self._aug_tokenizer = aug_tokenizer
        self._device = device
        self.num_aug = num_aug

    def startup_train_batch(self):
        self._aug_model.train()
        self._aug_model.zero_grad()

    def _augment_example(self, features, num_aug):
        init_ids, input_mask, segment_ids, _ = \
            (t.view(1, -1).to(self._device) for t in features)

        len = int(tx.sum(input_mask).item())
        if len >= 4:
            mask_idx = sorted(
                random.sample(list(range(1, len - 1)), max(len // 7, 2)))
        else:
            mask_idx = [1]

        init_ids[0][mask_idx] = \
            self._aug_tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        logits = self._aug_model(init_ids, segment_ids, input_mask)[0]

        # Get samples
        aug_probs_all = []
        for _ in range(num_aug):
            # might need a gumbel trick here in order to keep phi as variables
            probs = F.gumbel_softmax(logits, hard=False)
            aug_probs = torch.zeros_like(probs).scatter_(
                1, init_ids[0].unsqueeze(1), 1.)    # Todo: torch

            for t in mask_idx:
                aug_probs = tx.utils.pad_and_concat(
                    [aug_probs[:t], probs[t:t + 1], aug_probs[t + 1:]], dim=0)

            aug_probs_all.append(aug_probs)

        aug_probs = tx.utils.pad_and_concat([ap.unsqueeze(0) for ap in aug_probs_all], dim=0)

        return aug_probs

    def augment_example(self, features):
        aug_probs = self._augment_example(features, self.num_aug)

        _, input_mask, segment_ids, label_ids = \
            (t.to(self._device).unsqueeze(0) for t in features)
        num_aug_len = len(aug_probs)
        assert(num_aug_len==self.num_aug)
        input_mask_aug = tx.utils.pad_and_concat([input_mask] * self.num_aug, dim=0)
        segment_ids_aug = tx.utils.pad_and_concat([segment_ids] * self.num_aug, dim=0)
        label_ids_aug = tx.utils.pad_and_concat([label_ids] * self.num_aug, dim=0)
        return aug_probs, input_mask_aug, segment_ids_aug, label_ids_aug

    def update_meta_classifier(self, loss, classifier, classifier_optimizer):
        r"""
        equations:
        theta'(phi) = theta - \nabla_{theta} L_{train}(theta, phi)
        phi = phi - \nabla_{phi} L_{val}(theta'(phi))
        """

        # grads_theta(phi) = \nabla_{theta} L_{train}(theta, phi)
        grads_theta = self.calculate_grads(loss, classifier, classifier_optimizer)

        # meta model is used to calculate \nabla_{phi} L_{val}(theta'(phi)),
        # where it needs gradients applied to phi
        # meta model copies classifier and applies grad change to theta
        meta_model = MetaModule(classifier)

        # theta'(phi) = theta - grads_theta(phi)
        meta_model.update_parameters(grads_theta)

        return meta_model

    def update_phi(self):
        # phi = phi - \nabla_{phi} L_{val}(theta'(phi))
        self._aug_optimizer.step()

    def calculate_grads(self, loss, classifier, classifier_optimizer):
        grads = torch.autograd.grad(
            loss, [param for name, param in classifier.named_parameters()],
            create_graph=True)
        grads = {param: grads[i] for i, (name, param) in enumerate(
            classifier.named_parameters())}
        deltas = _adam_delta(classifier_optimizer, classifier, grads)
        return deltas

    def augment_batch(self, input_ids, input_mask, segment_ids, labels):
        self._aug_model.eval()

        aug_examples = []
        features = []
        for i in range(len(input_ids)):
            feature = (input_ids[i], input_mask[i], segment_ids[i], labels[i])
            features.append(feature)
            with torch.no_grad():
                aug_probs = self._augment_example(feature, num_aug=1)
                aug_examples.append(aug_probs)

        input_ids_or_probs, input_masks, segment_ids, label_ids = [tx.utils.pad_and_concat(
            [t[i].unsqueeze(0) for t in features], dim=0).to(
            self._device) for i in range(4)]

        num_aug = len(aug_examples[0])
        assert(num_aug==1)

        input_ids_or_probs_aug = []
        for i in range(num_aug):
            for aug_probs in aug_examples:
                input_ids_or_probs_aug.append(aug_probs[i:i + 1])
        input_ids_or_probs_aug = \
            tx.utils.pad_and_concat(input_ids_or_probs_aug, dim=0).to(self._device)

        inputs_onehot = torch.zeros_like(
            input_ids_or_probs_aug[:len(input_ids_or_probs)]).scatter_(
            2, input_ids_or_probs.unsqueeze(2), 1.)
        input_ids_or_probs = tx.utils.pad_and_concat(
            [inputs_onehot, input_ids_or_probs_aug], dim=0).to(self._device)

        segment_ids = \
            tx.utils.pad_and_concat([segment_ids] * (num_aug + 1), dim=0).to(self._device)
        input_masks = \
            tx.utils.pad_and_concat([input_masks] * (num_aug + 1), dim=0).to(self._device)
        label_ids = \
            tx.utils.pad_and_concat([label_ids] * (num_aug + 1), dim=0).to(self._device)

        return input_ids_or_probs, input_masks, segment_ids, label_ids


def _adam_delta(optimizer, model, grads):
    deltas = {}
    for group in optimizer.param_groups:
        for param in group['params']:
            grad = grads[param]
            state = optimizer.state[param]

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2 = group['betas']

            step = state['step'] + 1

            if group['weight_decay'] != 0:
                grad = grad + group['weight_decay'] * param.data

            exp_avg = exp_avg * beta1 + (1. - beta1) * grad
            exp_avg_sq = exp_avg_sq * beta2 + (1. - beta2) * grad * grad
            denom = exp_avg_sq.sqrt() + group['eps']

            bias_correction1 = 1. - beta1 ** step
            bias_correction2 = 1. - beta2 ** step
            step_size = group['lr'] * math.sqrt(
                bias_correction2) / bias_correction1

            deltas[param] = -step_size * exp_avg / denom

    param_to_name = {param: name for name, param in model.named_parameters()}

    return {param_to_name[param]: delta for param, delta in deltas.items()}
