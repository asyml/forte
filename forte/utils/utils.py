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
Utility functions
"""
from functools import wraps
from inspect import getfullargspec
from pydoc import locate
from typing import Dict, List, Optional, get_type_hints

from typing_inspect import is_union_type, get_origin

__all__ = [
    "get_full_module_name",
    "get_class_name",
    "get_class",
    "get_qual_name",
    "create_class_with_kwargs",
    "check_type",
]


def get_full_module_name(o, lower: bool = False) -> str:
    r"""Returns the full module and class name of an object ``o``. For example,
    for our :class:`OntonotesReader`, returns
    :class:'forte.data.readers.ontonotes_reader.OntonotesReader'.

    Args:
        o (object): An object class.
        lower (bool): Whether to lowercase the full module and class name.

    Returns:
         The full module and class name.
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


def get_class_name(o, lower: bool = False) -> str:
    r"""Returns the class name of an object ``o``.

    Args:
        o (object): An object class.
        lower (bool): Whether to lowercase the class name.

    Returns:
         The class name.
    """
    if not isinstance(o, type):
        o = o.__class__
    if lower:
        return o.__name__.lower()
    else:
        return o.__name__


def get_class(class_name: str,
              module_paths: Optional[List[str]] = None):
    r"""Returns the class based on class name.

    Args:
        class_name (str): Name or full path to the class.
        module_paths (list): Paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            ``class_name``. The first module in the list that contains the class
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
        if module_paths:
            raise ValueError(
                "Class not found in {}: {}".format(module_paths, class_name))
        else:
            raise ValueError(
                "Class not found in {}".format(class_name))

    return class_


def get_qual_name(o, lower: bool = False) -> str:
    r"""Returns the qualified name of an object ``o``.

    Args:
        o (object): An object class.
        lower (bool): Whether to lowercase the qualified class name.

    Returns:
         The qualified class name.
    """
    if not isinstance(o, type):
        o = o.__class__
    if lower:
        return o.__qualname__.lower()
    else:
        return o.__qualname__


def create_class_with_kwargs(class_name: str, class_args: Dict):
    r"""Create class with the given arguments.

    Args:
        class_name (str): Class name.
        class_args (Dict): Class arguments.

    Returns:
        An object with class of type `class_name`.
    """
    cls = get_class(class_name)
    if not class_args:
        class_args = {}
    obj = cls(**class_args)

    return obj


def check_type(obj, tp):
    if is_union_type(tp):
        return any(check_type(obj, a) for a in tp.__args__)
    else:
        origin = get_origin(tp)
        if origin is None or origin == tp:
            return isinstance(obj, tp)
        else:
            return check_type(obj, origin)


def validate_input(func, **kwargs):
    hints = get_type_hints(func)

    # iterate all type hints
    for attr_name, attr_type in hints.items():
        if attr_name == 'return':
            continue

        if not isinstance(kwargs[attr_name], attr_type):
            raise TypeError(
                f'{attr_name} should be of type {attr_type}, '
                f'got type {type(kwargs[attr_name])}'
            )


def type_check(func):
    @wraps(func)
    def wrapped_decorator(*args, **kwargs):
        # translate *args into **kwargs
        func_args = getfullargspec(func)[0]
        kwargs.update(dict(zip(func_args, args)))
        validate_input(func, **kwargs)
        return func(**kwargs)

    return wrapped_decorator
