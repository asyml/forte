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

# pylint: disable=non-parent-init-called
# pylint: disable=super-init-not-called
"""
A magic model that registers the parameter of a pytorch nn module
and performs memory-efficient parameter updates locally.
"""

import copy
from typing import Dict
import torch
from torch import nn

try:
    import texar.torch as tx
except ImportError as e:
    raise ImportError(
        " `texar-pytorch` is not installed correctly."
        " Please refer to [extra requirement for aug wrapper](pip install forte[models])"
        " for more information. "
    ) from e

__all__ = ["MetaModule", "TexarBertMetaModule"]


class MetaModule(nn.ModuleList):
    # pylint: disable=line-too-long
    r"""A class extending :class:`torch.nn.ModuleList`
    that registers the parameters of a :class:`torch.nn.Module`
    and performs memory-efficient parameter updates locally.

    This code is adapted from:
    https://github.com/tanyuqian/learning-data-manipulation/blob/master/magic_module.py

    It implements the calculation:
    :math:`L(\theta - \nabla_{\theta} L_{train}(\theta, \phi))`.

    Args:
        module: A :class:`torch.nn.Module`.

    This class can be used for simple input module, whose sub-modules don't
    contain other helper functions or attributes that do not belong to this
    class to perform their :meth:`forward`.

    Otherwise, since :meth:`forward` calls the input module's :meth:`forward`,
    in order to perform :meth:`forward` of the sub-modules of the input module
    correctly, this class needs to extend those sub-modules that define
    the methods needed for their :meth:`forward`, so that it inherits their
    methods to perform the sub-module's :meth:`forward`.

    For example, if the input module is
    :class:`~texar.torch.modules.BERTClassifier`,
    :meth:`_get_noise_shape`, :meth:`_split_heads`, :meth:`_combine_heads`
    from its sub-modules (E.g. :class:`~texar.torch.modules.BERTEncoder`)
    are needed to be
    exposed in this class to perform their :meth:`forward`. Please refer to
    :class:`TexarBertMetaModule` for instructions on creating a subclass from
    this one for a specific input module.
    """

    def __init__(self, module: nn.Module):
        nn.Module.__init__(self)
        self._type = type(module)

        for key, value in module._parameters.items():
            if value is not None:
                self.register_parameter("_origin_" + key, value)
                self.register_buffer(key, value.data)
            else:
                self.register_buffer(key, None)

        for key, value in module._buffers.items():
            self.register_buffer(key, copy.deepcopy(value))

        # Recursively create MetaModule.
        for key, value in module._modules.items():
            # type(self) is the real class object
            # it can be MetaModule(value), or it can be its subclass,
            # e.g. TexarBertMetaModule(value)
            self.add_module(key, type(self)(value))

        for key, value in module.__dict__.items():
            if (
                key not in self.__dict__
                and key not in self._buffers
                and key not in self._modules
            ):
                self.__setattr__(key, value)

    def forward(self, *args, **kwargs):
        return self._type.forward(self, *args, **kwargs)

    def update_params(self, deltas: Dict[str, torch.Tensor]):
        sub_params: Dict[str, torch.Tensor] = {}
        for key, delta in deltas.items():
            if "." not in key:
                self._buffers[key] = self._buffers[key] + delta
            else:
                attr = key.split(".")[0]
                if attr not in sub_params:
                    sub_params[attr] = {}
                sub_params[attr][".".join(key.split(".")[1:])] = delta
        for key, value in sub_params.items():
            self._modules[key].update_params(value)


class TexarBertMetaModule(
    MetaModule, tx.modules.EmbedderBase, tx.modules.MultiheadAttentionEncoder
):
    r"""A subclass that extends :class:`MetaModule` to do parameter updates
    locally for texar-pytorch Bert related modules.
    E.g. :class:`texar.torch.modules.BERTClassifier`

    Please refer to its base class :class:`MetaModule` for more details.

    Args:
        module: A :class:`torch.nn.Module`.

    This class extends :class:`~texar.torch.modules.EmbedderBase` and
    :class:`~texar.torch.modules.MultiheadAttentionEncoder`, such that it
    inherits their methods that are needed to perform :meth:`forward` of
    the modules that utilizes these methods,
    E.g. :class:`~texar.torch.modules.BERTEncoder`,

    Some notes of the order of the base classes that this class extends:

    `MetaModule` should be the first one, so that its :meth:`forward` will
    call :meth:`MetaModule.forward` instead of the :meth:`forward` of the other
    base classes, such as
    :func:`texar.torch.modules.MultiheadAttentionEncoder.forward`.
    If `MetaModule` is not the first one, then a :meth:`forward` should be
    defined in this class, such that it is called correctly.

    Example:

        .. code-block:: python

            def forward(self, *args, **kwargs):
                return MetaModule.forward(self, *args, **kwargs)

    """

    def __init__(self, module: nn.Module):
        MetaModule.__init__(self, module)
