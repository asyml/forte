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
Util functions
"""
from typing import Dict, Optional
from pydoc import locate
import yaml

from texar.torch import HParams

__all__ = [
    "get_full_module_name",
    "get_class_name",
    "get_class",
    "get_qual_name",
    "create_class_with_kwargs"
]


def get_full_module_name(o, lower=False):
    """
    Returns the full module and class name of an object o.
    For example, for our :class: OntonotesReader, returns
    'nlp.forte.data.readers.ontonotes_reader.OntonotesReader'.
    """
    if not isinstance(o, type):
        o = o.__class__
    module = o.__module__
    if module is None or module == str.__class__.__module__:
        return o.__name__
    name = module + '.' + o.__name__
    if lower:
        return name.lower()
    else:
        return name


def get_class_name(o, lower=False):
    """
    Returns the class name of an object o.
    """
    if not isinstance(o, type):
        o = o.__class__
    if lower:
        return o.__name__.lower()
    else:
        return o.__name__


def get_class(class_name, module_paths=None):
    """Returns the class based on class name.

    Args:
        class_name (str): Name or full path to the class.
        module_paths (list): Paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            `class_name`. The first module in the list that contains the class
            is used.

    Returns:
        The target class.

    Raises:
        ValueError: If class is not found based on :attr:`class_name` and
            :attr:`module_paths`.
    """
    class_ = locate(class_name)
    if (class_ is None) and (module_paths is not None):
        for module_path in module_paths:
            class_ = locate('.'.join([module_path, class_name]))
            if class_ is not None:
                break

    if class_ is None:
        raise ValueError(
            "Class not found in {}: {}".format(module_paths, class_name))

    return class_


def get_qual_name(o, lower=False):
    """
    Returns the qualified name of an object o.
    """
    if not isinstance(o, type):
        o = o.__class__
    if lower:
        return o.__qualname__.lower()
    else:
        return o.__qualname__


def create_class_with_kwargs(
        class_name: str, class_args: Dict, h_params: Optional[Dict] = None):
    cls = get_class(class_name)
    if not class_args:
        class_args = {}
    obj = cls(**class_args)

    if h_params is None:
        h_params = {}

    p_params: Dict = {}

    if "config_path" in h_params and not h_params["config_path"] is None:
        filebased_hparams = yaml.safe_load(open(h_params["config_path"]))
    else:
        filebased_hparams = {}
    p_params.update(filebased_hparams)

    p_params.update(h_params.get("overwrite_configs", {}))
    default_processor_hparams = cls.default_hparams()

    processor_hparams = HParams(p_params,
                                default_processor_hparams)

    return obj, processor_hparams
