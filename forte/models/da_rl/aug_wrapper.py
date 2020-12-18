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
A wrapper adding data augmentation to a Bert model with arbitrary tasks.
"""

import random
import math
from typing import Tuple, Dict, Generator
import texar.torch as tx
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Optimizer

from forte.models.da_rl.magic_model import MetaModule


__all__ = [
    "MetaAugmentationWrapper"
]


class MetaAugmentationWrapper:
    # pylint: disable=line-too-long
    r"""
    A wrapper adding data augmentation to a Bert model with arbitrary tasks.
    This is used to perform reinforcement learning for joint data augmentation
    learning and model training.

    See: https://arxiv.org/pdf/1910.12795.pdf

    This code is adapted from:
    https://github.com/tanyuqian/learning-data-manipulation/blob/master/augmentation/generator.py

    Let :math:`\theta` be the parameters of the downstream (classifier) model.
    Let :math:`\phi` be the parameters of the augmentation model.
    Equations to update :math:`\phi`:

    .. math::

        \theta'(\phi) = \theta - \nabla_{\theta} L_{train}(\theta, \phi)

        \phi = \phi - \nabla_{\phi} L_{val}(\theta'(\phi))

    Args:
        augmentation_model:
            A Bert-based model for data augmentation.
            Eg. BertForMaskedLM.
            Model requirement: masked language modeling, the output logits
            of this model is of shape `[batch_size, seq_length, token_size]`.
        augmentation_optimizer:
            An optimizer that is associated with `augmentation_model`.
            Eg. Adam optim
        input_mask_ids:
            Bert token id of `'[MASK]'`. This is used to randomly mask out
            tokens from the input sentence during training.
        device:
            The CUDA device to run the model on.
        num_aug:
            The number of samples from the augmentation model
            for every augmented training instance.

    Example usage:

        .. code-block:: python

            aug_wrapper = MetaAugmentationWrapper(
                aug_model, aug_optim, mask_id, device, num_aug)
            for batch in training_data:
                # Train augmentation model params.
                aug_wrapper.reset_model()
                for instance in batch:
                    # Augmented example with params phi exposed
                    aug_instance_features = \
                        aug_wrapper.augment_example(instance_features)
                    # Model is the downstream Bert model.
                    model.zero_grad()
                    loss = model(aug_instance_features)
                    meta_model = MetaModule(model)
                    meta_model = aug_wrapper.update_meta_classifier(
                        meta_model, loss, model, optim)

                    # Compute grads of aug model on validation data
                    for val_batch in validation_data:
                        val_loss = meta_model(val_batch_features)
                        val_loss = val_loss / num_training_instance / num_aug \
                            / num_val_batch
                        val_loss.backward()
                # update augmentation model params.
                aug_wrapper.update_phi()

                # train classifier with augmented batch
                aug_batch_features = aug_wrapper.augment_batch(batch_features)
                optim.zero_grad()
                loss = model(aug_batch_features)
                loss.backward()
                optim.step()

    """

    def __init__(self, augmentation_model: nn.Module,
                 augmentation_optimizer: Optimizer,
                 input_mask_ids: int, device: torch.device, num_aug: int):
        self._aug_model = augmentation_model
        self._aug_optimizer = augmentation_optimizer
        self._input_mask_ids = input_mask_ids
        self._device = device
        self._num_aug = num_aug

    def reset_model(self):
        self._aug_model.train()
        self._aug_model.zero_grad()

    def _augment_instance(self, features: Tuple[torch.Tensor, ...],
                          num_aug: int) -> torch.Tensor:
        r"""Augment a training instance. Randomly mask out some tokens in the
        input sentence and use the logits of the augmentation model as the
        augmented bert token soft embedding.

        Args:
            features: A tuple of Bert features of one training instance.
                (input_ids, input_mask, segment_ids, label_ids).

                `input_ids` is a tensor of Bert token ids.
                It has shape `[seq_len]`.

                `input_mask` is a tensor of shape `[seq_len]` with 1 indicating
                without mask and 0 with mask.

                `segment_ids` is a tensor of shape `[seq_len]`.
                `label_ids` is a tensor of shape `[seq_len]`.
            num_aug: The number of samples from the augmentation model.

        Returns:
            aug_probs: A tensor of shape `[num_aug, seq_len, token_size]`.
                It is the augmented bert token soft embedding.
        """

        feature: Generator[torch.Tensor, torch.Tensor, torch.Tensor] = \
            (t.view(1, -1).to(self._device) for t in features)
        init_ids, input_mask, segment_ids, _ = feature

        len = int(torch.sum(input_mask).item())

        if len >= 4:
            mask_idx = sorted(
                random.sample(list(range(1, len - 1)), max(len // 7, 2)))
        else:
            mask_idx = [1]

        init_ids[0][mask_idx] = self._input_mask_ids

        logits = self._aug_model(init_ids,
                                 token_type_ids=segment_ids,
                                 attention_mask=input_mask)[0]

        # Get samples
        aug_probs_all = []
        for _ in range(num_aug):
            # Need a gumbel trick here in order to keep phi as variables.
            # Enable efficient gradient propagation through theta' to phi.
            probs = F.gumbel_softmax(logits.squeeze(0), hard=False)
            aug_probs = torch.zeros_like(probs).scatter_(
                1, init_ids[0].unsqueeze(1), 1.)

            for t in mask_idx:
                aug_probs = tx.utils.pad_and_concat(
                    [aug_probs[:t], probs[t:t + 1], aug_probs[t + 1:]], axis=0)

            aug_probs_all.append(aug_probs)

        aug_probs = tx.utils.pad_and_concat(
            [ap.unsqueeze(0) for ap in aug_probs_all], axis=0)

        return aug_probs

    def augment_instance(self, features: Tuple[torch.Tensor, ...]) \
            -> Tuple[torch.Tensor, ...]:
        r"""Augment a training instance.

        Args:
            features: A tuple of Bert features of one training instance.
                (input_ids, input_mask, segment_ids, label_ids).

                `input_ids` is a tensor of Bert token ids.
                It has shape `[seq_len]`.

                `input_mask` is a tensor of shape `[seq_len]` with 1 indicating
                without mask and 0 with mask.

                `segment_ids` is a tensor of shape `[seq_len]`.
                `label_ids` is a tensor of shape `[seq_len]`.

        Returns:
            A tuple of Bert features of augmented training instances.
            (input_probs_aug, input_mask_aug, segment_ids_aug, label_ids_aug).

            `input_probs_aug` is a tensor of soft Bert embeddings,
            distributions over vocabulary.
            It has shape `[num_aug, seq_len, token_size]`.
            It keeps :math:`\phi` as variable so that after passing it as an
            input to the classifier, the gradients of :math:`\theta` will
            also apply to :math:`\phi`.

            `input_mask_aug` is a tensor of shape `[num_aug, seq_len]`, it
            concatenates `num_aug` the input `input_mask` so that it
            corresponds to the mask of each token in `input_probs_aug`.

            `segment_ids_aug` is a tensor of shape `[num_aug, seq_len]`, it
            concatenates `num_aug` the input `segment_ids` so that it
            corresponds to the token type of each token in `input_probs_aug`.

            `label_ids_aug` is a tensor of shape `[num_aug, seq_len]`, it
            concatenates `num_aug` the input `label_ids` so that it corresponds
            to the label of each token in `input_probs_aug`.
        """

        aug_probs = self._augment_instance(features, self._num_aug)

        _, input_mask, segment_ids, label_ids = \
            (t.to(self._device).unsqueeze(0) for t in features)
        input_mask_aug = tx.utils.pad_and_concat(
            [input_mask] * self._num_aug, axis=0)
        segment_ids_aug = tx.utils.pad_and_concat(
            [segment_ids] * self._num_aug, axis=0)
        label_ids_aug = tx.utils.pad_and_concat(
            [label_ids] * self._num_aug, axis=0)

        return aug_probs, input_mask_aug, segment_ids_aug, label_ids_aug

    def augment_batch(self, batch_features: Tuple[torch.Tensor, ...]) \
            -> Tuple[torch.Tensor, ...]:
        r"""Augment a batch of training instances. Append augmented instances
        to the input instances.

        Args:
            batch_features: A tuple of Bert features of a batch training
                instances. (input_ids, input_mask, segment_ids, label_ids).

                `input_ids` is a tensor of Bert token ids.
                It has shape `[batch_size, seq_len]`.

                `input_mask`, `segment_ids`, `label_ids` are all tensors of
                shape `[batch_size, seq_len]`.

        Returns:
            A tuple of Bert features of augmented training instances.
            (input_probs_aug, input_mask_aug, segment_ids_aug, label_ids_aug).

            `input_probs_aug` is a tensor of soft Bert embeddings,
            It has shape `[batch_size * 2, seq_len, token_size]`.

            `input_mask_aug` is a tensor of shape `[batch_size * 2, seq_len]`,
            it concatenates two input `input_mask`, the first one corresponds to the
            mask of the tokens in the original bert instance, the second one
            corresponds to the mask of the augmented bert instance.

            `segment_ids_aug` is a tensor of shape `[batch_size * 2, seq_len]`,
            it concatenates two input `segment_ids`, the first one corresponds
            to the segment id of the tokens in the original bert instance, the
            second one corresponds to the segment id of the
            augmented bert instance.

            `label_ids_aug` is a tensor of shape `[batch_size * 2, seq_len]`,
            it concatenates two input `label_ids`, the first one corresponds
            to the labels of the original bert instance, the second one
            corresponds to the labels of the augmented bert instance.
        """
        input_ids, input_mask, segment_ids, labels = batch_features
        self._aug_model.eval()

        aug_instances = []
        features = []
        num_instance = len(input_ids)
        for i in range(num_instance):
            feature = (input_ids[i], input_mask[i], segment_ids[i], labels[i])
            features.append(feature)
            with torch.no_grad():
                aug_probs = self._augment_instance(feature, num_aug=1)
                aug_instances.append(aug_probs)

        input_ids_or_probs, input_masks, segment_ids, label_ids = \
            [tx.utils.pad_and_concat(
                [t[i].unsqueeze(0) for t in features], axis=0).to(
                self._device) for i in range(4)]

        num_aug = len(aug_instances[0])

        input_ids_or_probs_aug = []
        for i in range(num_aug):
            for aug_probs in aug_instances:
                input_ids_or_probs_aug.append(aug_probs[i:i + 1])
        input_ids_or_probs_aug = tx.utils.pad_and_concat(
            input_ids_or_probs_aug, axis=0).to(self._device)

        inputs_onehot = torch.zeros_like(
            input_ids_or_probs_aug[:len(input_ids_or_probs)]).scatter_(
            2, input_ids_or_probs.unsqueeze(2), 1.)
        input_probs_aug = tx.utils.pad_and_concat(
            [inputs_onehot, input_ids_or_probs_aug], axis=0).to(self._device)

        input_mask_aug = tx.utils.pad_and_concat(
            [input_masks] * (num_aug + 1), axis=0).to(self._device)
        segment_ids_aug = tx.utils.pad_and_concat(
            [segment_ids] * (num_aug + 1), axis=0).to(self._device)
        label_ids_aug = tx.utils.pad_and_concat(
            [label_ids] * (num_aug + 1), axis=0).to(self._device)

        return input_probs_aug, input_mask_aug, segment_ids_aug, label_ids_aug

    def eval_batch(self, batch_features: Tuple[torch.Tensor, ...]) \
            -> torch.FloatTensor:
        r"""Evaluate a batch of training instances.

        Args:
            batch_features: A tuple of Bert features of a batch training
                instances. (input_ids, input_mask, segment_ids, label_ids).

                `input_ids` is a tensor of Bert token ids.
                It has shape `[batch_size, seq_len]`.

                `input_mask`, `segment_ids`, `label_ids` are all tensors of
                shape `[batch_size, seq_len]`.

        Returns:
            The masked language modeling loss of one evaluation batch.
            It is a `torch.FloatTensor` of shape `[1,]`.
        """
        self._aug_model.eval()
        batch = tuple(t.to(self._device) for t in batch_features)
        input_ids, input_mask, segment_ids, labels = batch
        loss = self._aug_model(input_ids, token_type_ids=segment_ids,
                               attention_mask=input_mask, labels=labels)[0]
        return loss

    def update_meta_model(self, meta_model: MetaModule, loss: torch.Tensor,
                          model: nn.Module, optimizer: Optimizer) \
            -> MetaModule:
        r"""Update the parameters within the `MetaModel`
        according to the downstream model loss.

        `MetaModel` is used to calculate
        :math:`\nabla_{\phi} L_{val}(\theta'(\phi))`,
        where it needs gradients applied to :math:`\phi`.

        Perform parameter updates in this function, and later applies gradient
        change to :math:`\theta` and :math:`\phi` using validation data.

        Args:
            meta_model: A meta model whose parameters will be updated in-place
                by the deltas calculated from the input `loss`.
            loss: The loss of the downstream model that have taken
                the augmented training instances as input.
            model: The downstream Bert model.
            optimizer: The optimizer that is associated with the `model`.

        Returns:
            The same input `meta_model` with the updated parameters.
        """

        # grads_theta(phi) = \nabla_{theta} L_{train}(theta, phi)
        grads_theta = self._calculate_grads(loss, model, optimizer)

        # theta'(phi) = theta - grads_theta(phi)
        meta_model.update_params(grads_theta)

        return meta_model

    @staticmethod
    def _calculate_grads(loss: torch.Tensor, model: nn.Module,
                         optimizer: Optimizer) -> Dict[str, torch.Tensor]:
        grads = torch.autograd.grad(
            loss, [param for name, param in model.named_parameters()],
            create_graph=True)
        grads = {param: grads[i] for i, (name, param) in enumerate(
            model.named_parameters())}

        if isinstance(optimizer, tx.core.BertAdam):
            deltas = _texar_bert_adam_delta(grads, model, optimizer)
        else:
            deltas = _torch_adam_delta(grads, model, optimizer)

        return deltas

    def update_phi(self):
        # L_{val}(theta'(phi))
        # apply gradients to phi

        # phi = phi - \nabla_{phi} L_{val}(theta'(phi))
        self._aug_optimizer.step()


def _texar_bert_adam_delta(grads: Dict[nn.parameter.Parameter, torch.Tensor],
                           model: nn.Module,
                           optimizer: Optimizer) -> Dict[str, torch.Tensor]:
    # pylint: disable=line-too-long
    r"""Compute parameter delta function for texar-pytorch
    core.BertAdam optimizer.

    This function is adapted from:
    https://github.com/asyml/texar-pytorch/blob/master/texar/torch/core/optimization.py#L398
    """
    assert isinstance(optimizer, tx.core.BertAdam)

    deltas = {}
    for group in optimizer.param_groups:
        for param in group['params']:
            grad = grads[param]
            state = optimizer.state[param]

            if len(state) == 0:
                # Exponential moving average of gradient values
                state['next_m'] = torch.zeros_like(param.data)
                # Exponential moving average of squared gradient values
                state['next_v'] = torch.zeros_like(param.data)

            exp_avg, exp_avg_sq = state['next_m'], state['next_v']

            beta1, beta2 = group['betas']

            if group['weight_decay'] != 0:
                grad = grad + group['weight_decay'] * param.data

            exp_avg = exp_avg * beta1 + (1. - beta1) * grad
            exp_avg_sq = exp_avg_sq * beta2 + (1. - beta2) * grad * grad
            denom = exp_avg_sq.sqrt() + group['eps']

            step_size = group['lr']

            deltas[param] = -step_size * exp_avg / denom

    param_to_name = {param: name for name, param in model.named_parameters()}

    return {param_to_name[param]: delta for param, delta in deltas.items()}


def _torch_adam_delta(grads: Dict[nn.parameter.Parameter, torch.Tensor],
                      model: nn.Module,
                      optimizer: Optimizer) -> Dict[str, torch.Tensor]:
    r"""Compute parameter delta function for Torch Adam optimizer.
    """
    assert issubclass(type(optimizer), Optimizer)

    deltas = {}
    for group in optimizer.param_groups:
        for param in group['params']:
            grad = grads[param]
            state = optimizer.state[param]

            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(param.data)
                state['exp_avg_sq'] = torch.zeros_like(param.data)
                state['step'] = 0

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
