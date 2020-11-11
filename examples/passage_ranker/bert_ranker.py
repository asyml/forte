# Copyright 2019 The Forte Authors. All Rights Reserved.
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
BERT encoder and BERT classifier. BERT classifier takes in fine-tuned BERT
encoder.
"""
import os
from typing import Optional, cast

import torch
from torch.nn import Parameter
from texar.torch.modules.pretrained import PretrainedBERTMixin
from texar.torch.modules.encoders import BERTEncoder as TxBERTEncoder
from texar.torch.modules.classifiers import BERTClassifier as TxBERTClassifier

from forte.common.configuration import Config

__all__ = [
    "BERTEncoder",
    "BERTClassifier"
]


class BERTClassifier(TxBERTClassifier, PretrainedBERTMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        hparams = self.default_hparams()
        hparams.update(kwargs)

        self.load_pretrained_config(hparams=hparams)

        super().__init__(pretrained_model_name=self.pretrained_model_name,
                         cache_dir=self.cache_dir)

        self.init_pretrained_weights(prefix=self._hparams.prefix)

    def default_hparams(self):
        params = super().default_hparams()
        params.update(self.kwargs)
        return params

    def load_pretrained_config(self,
                               pretrained_model_name: Optional[str] = None,
                               cache_dir: Optional[str] = None,
                               hparams=None):
        r"""Load paths and configurations of the pre-trained model.

        Args:
            pretrained_model_name (optional): A str with the name
                of a pre-trained model to load. If `None`, will use the model
                name in :attr:`hparams`.
            cache_dir (optional): The path to a folder in which the
                fine-tuned model is present.
            hparams (dict or HParams, optional): Hyperparameters. Missing
                hyperparameter will be set to default values. See
                :meth:`default_hparams` for the hyperparameter structure
                and default values.
        """

        self.pretrained_model_name = hparams['pretrained_model_name'] \
            if pretrained_model_name is None else pretrained_model_name

        rel_dir = hparams['model_dir'] if cache_dir is None else cache_dir
        self.cache_dir = os.path.join(os.path.dirname(__file__), rel_dir)

        if self.pretrained_model_name is None or self.cache_dir is None:
            raise ValueError("Pre-trained model name and directory should"
                             "be defined in the fine tuned BERT model.")

        self.pretrained_model_dir = os.path.join(self.cache_dir,
                                                 self.pretrained_model_name)

        pretrained_model_hparams = self._transform_config(
            self.pretrained_model_name, self.pretrained_model_dir)

        super_params = self.default_hparams()
        if 'prefix' not in super_params:
            super_params["prefix"] = '_encoder.encoder.'
        self._hparams = Config(pretrained_model_hparams, super_params)

    def _init_from_checkpoint(self, pretrained_model_name: str,
                              cache_dir: str, **kwargs):
        # pylint: disable=import-outside-toplevel
        try:
            import numpy as np
            import tensorflow as tf
        except ImportError:
            print("Loading TensorFlow models in PyTorch requires installing "
                  "TensorFlow. Please see https://www.tensorflow.org/install/ "
                  "for installation instructions.")
            raise

        py_prefix = "encoder." if 'prefix' not in kwargs else kwargs['prefix']

        global_tensor_map = {
            'bert/embeddings/word_embeddings': 'word_embedder._embedding',
            'bert/embeddings/token_type_embeddings':
                'segment_embedder._embedding',
            'bert/embeddings/position_embeddings':
                'position_embedder._embedding',
            'bert/embeddings/LayerNorm/beta':
                'encoder.input_normalizer.bias',
            'bert/embeddings/LayerNorm/gamma':
                'encoder.input_normalizer.weight',
        }
        layer_tensor_map = {
            "attention/self/key/bias": "self_attns.{}.K_dense.bias",
            "attention/self/query/bias": "self_attns.{}.Q_dense.bias",
            "attention/self/value/bias": "self_attns.{}.V_dense.bias",
            "attention/output/dense/bias": "self_attns.{}.O_dense.bias",
            "attention/output/LayerNorm/gamma": "poswise_layer_norm.{}.weight",
            "attention/output/LayerNorm/beta": "poswise_layer_norm.{}.bias",
            "intermediate/dense/bias": "poswise_networks.{}._layers.0.bias",
            "output/dense/bias": "poswise_networks.{}._layers.2.bias",
            "output/LayerNorm/gamma": "output_layer_norm.{}.weight",
            "output/LayerNorm/beta": "output_layer_norm.{}.bias",
        }
        layer_transpose_map = {
            "attention/self/key/kernel": "self_attns.{}.K_dense.weight",
            "attention/self/query/kernel": "self_attns.{}.Q_dense.weight",
            "attention/self/value/kernel": "self_attns.{}.V_dense.weight",
            "attention/output/dense/kernel": "self_attns.{}.O_dense.weight",
            "intermediate/dense/kernel": "poswise_networks.{}._layers.0.weight",
            "output/dense/kernel": "poswise_networks.{}._layers.2.weight",
        }
        pooler_map = {
            'bert/pooler/dense/bias': 'pooler.0.bias',
            'bert/pooler/dense/kernel': 'pooler.0.weight'
        }
        output_map = {
            'output_weights': '_logits_layer.weight',
            'output_bias': '_logits_layer.bias'
        }
        tf_path = os.path.abspath(os.path.join(
            cache_dir, self._MODEL2CKPT[pretrained_model_name]))

        # Load weights from TF model
        init_vars = tf.train.list_variables(tf_path)
        tfnames, arrays = [], []
        for name, _ in init_vars:
            array = tf.train.load_variable(tf_path, name)
            tfnames.append(name)
            arrays.append(array.squeeze())

        idx = 0
        for name, array in zip(tfnames, arrays):
            if name.startswith('cls') or name == 'global_step' or \
                    name.endswith('adam_m') or name.endswith('adam_v'):
                # ignore those variables begin with cls
                # ignore 'global_step' variable
                # ignore optimizer state variable
                continue

            if name in global_tensor_map:
                v_name = '_encoder.' + global_tensor_map[name]
                pointer = self._name_to_variable(v_name)
                assert pointer.shape == array.shape
                pointer.data = torch.from_numpy(array)
                idx += 1
            elif name in pooler_map:
                v_name = '_encoder.' + pooler_map[name]
                pointer = self._name_to_variable(v_name)
                if name.endswith('bias'):
                    assert pointer.shape == array.shape
                    pointer.data = torch.from_numpy(array)
                    idx += 1
                else:
                    array_t = np.transpose(array)
                    assert pointer.shape == array_t.shape
                    pointer.data = torch.from_numpy(array_t)
                    idx += 1
            elif name in output_map:
                v_name = output_map[name]
                pointer = self._name_to_variable(v_name)
                if name.endswith('bias'):
                    assert pointer.shape == array.shape
                    pointer.data = torch.from_numpy(array)
                    idx += 1
                else:
                    assert pointer.shape == array.shape
                    pointer.data = torch.from_numpy(array)
                    idx += 1
            else:
                # here name is the TensorFlow variable name
                name_tmp = name.split("/")
                # e.g. layer_
                layer_no = name_tmp[2][6:]
                name_tmp = "/".join(name_tmp[3:])
                if name_tmp in layer_tensor_map:
                    v_name = layer_tensor_map[name_tmp].format(layer_no)
                    pointer = self._name_to_variable(py_prefix + v_name)
                    assert pointer.shape == array.shape
                    pointer.data = torch.from_numpy(array)
                elif name_tmp in layer_transpose_map:
                    v_name = layer_transpose_map[name_tmp].format(layer_no)
                    pointer = self._name_to_variable(py_prefix + v_name)
                    array_t = np.transpose(array)
                    assert pointer.shape == array_t.shape
                    pointer.data = torch.from_numpy(array_t)
                else:
                    raise NameError(f"Variable with name '{name}' not found")
                idx += 1

    def _name_to_variable(self, name: str) -> Parameter:
        """
            Cast return value of method of the super class
            `texar.torch.module.PretrainedBertMixin._name_to_variable` to avoid
            type errors
        """
        return cast(Parameter, super()._name_to_variable(name))


class BERTEncoder(TxBERTEncoder):
    # pylint: disable=unused-argument
    def load_pretrained_config(self,
                               pretrained_model_name: Optional[str] = None,
                               cache_dir: Optional[str] = None,
                               hparams=None):
        # pylint: disable=attribute-defined-outside-init
        self.pretrained_model_name = pretrained_model_name
        self.cache_dir = cache_dir

    def init_pretrained_weights(self):
        pass

    @staticmethod
    def default_hparams():
        params = BERTEncoder.default_hparams()
        params['model_dir'] = None
        return params
