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
from texar.torch.modules.classifiers.bert_classifier import BERTClassifier
import copy
import torch


class MetaAugmentationWrapper:
    '''
    a wrapper adding data augmentation to a model with arbitrary tasks
    see: https://arxiv.org/pdf/1910.12795.pdf

    let theta be the parameters of the classifer model
    let phi be the parameters of the augmentation model

    '''
    def __init__(self, augmentation_model):
        self.generator = augmentation_model
        self.optimizer_phi = augmentation_model.get_optimizer()

    def augment(self, classifier, batch):
        r"""
        args: classifer as model_theta
        return: augmented_batch_examples
        """
        self.update_phi(classifier, batch)
        return self.generator(batch)

    def update_phi(self, classifier, train_batch):
        r"""
        equations:
        theta'(phi) = theta - \nabla_{theta} L_{train}(theta, phi)
        phi = phi - \nabla_{phi} L_{val}(theta'(phi))
        """

        # grads_theta(phi) = \nabla_{theta} L_{train}(theta, phi)
        # might need a gumbel trick here in order to keep phi as variables
        augmented_batch = self._generator(train_batch)  # with phi as var   # Todo

        grads_theta = calculate_grads(augmented_batch, classifier)   # see example below

        # meta model is used to calculate \nabla_{phi} L_{val}(theta'(phi)),
        # where it needs gradients applied to phi
        # meta model copies classifier and applies grad change to theta
        meta_model = MetaModule(classifier)

        # theta'(phi) = theta - grads_theta(phi)
        meta_model.update_parameters(grads_theta)

        for val_batch in val_data:
            # L_{val}(theta'(phi))
            val_loss = meta_model(val_batch)

            # apply gradients to phi
            val_loss.backward()

            # phi = phi - \nabla_{phi} L_{val}(theta'(phi))
            self.optimizer_phi.step()



def calculate_grads(augmented_batch, classifier):
    return


class MetaModule(tx.modules):
    '''
    input: a pytorch module

    implement the calculation:
    L(theta - \nabla_{theta} L_{train}(theta, phi))
    after the calculation, we need phi is derivable

    there is an example code for this class here:
    https://github.com/tanyuqian/learning-data-manipulation/blob/master/magic_module.py
    '''

    def __init__(self, module):
        tx.modules.__init__(self)
        self._type = type(module)

        for key, value in module._parameters.items():
            if value is not None:
                self.register_parameter('_origin_' + key, value)
                self.register_buffer(key, value.data)
            else:
                self.register_buffer(key, None)

        for key, value in module._buffers.items():
            self.register_buffer(key, copy.deepcopy(value))

        for key, value in module._modules.items():
            self.add_module(key, MetaModule(value))

        for key, value in module.__dict__.items():
            if (not key in self.__dict__) and\
                    (not key in self._buffers) and\
                    (not key in self._modules):
                self.__setattr__(key, value)

    def update_params(self, deltas):
        sub_params = {}
        for key, delta in deltas.items():
            if not ('.' in key):
                self._buffers[key] = self._buffers[key] + delta
            else:
                attr = key.split('.')[0]
                if not (attr in sub_params):
                    sub_params[attr] = {}
                sub_params[attr]['.'.join(key.split('.')[1:])] = delta
        for key, value in sub_params.items():
            self._modules[key].update_params(value)


# An example of classifier calculate_grads
# from https://github.com/tanyuqian/learning-data-manipulation/blob/master/augmentation/classifier.py

class Classifier:
    def __init__(self):
        self._model = BERTClassifier('bert-base-uncased')

    def calculate_grads(self):
        self._model.zero_grad()
        loss = self._model(
            aug_probs, segment_ids_aug, input_mask_aug, label_ids_aug,
            use_input_probs=True)
        grads = torch.autograd.grad(
            loss, [param for name, param in self._model.named_parameters()],
            create_graph=True)

        grads = {param: grads[i] for i, (name, param) in enumerate(
            self._model.named_parameters())}

        deltas = _adam_delta(self._optimizer, self._model, grads)
        return deltas

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