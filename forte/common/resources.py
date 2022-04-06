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

from collections.abc import KeysView
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pickle

__all__ = ["Resources"]

SerializeDict = Dict[str, Callable[[Any, Union[str, Path]], None]]
DeserializeDict = Dict[str, Callable[[Union[str, Path]], None]]


class Resources:
    r"""The :class:`~forte.common.resources.Resources` object is a global
    registry used in the pipeline.
    Objects defined as :class:`~forte.common.resources.Resources` will be
    passed on to the processors in the
    pipeline for initialization.
    """

    def __init__(self, **kwargs):
        self._resources = {}
        self.update(**kwargs)

    def save(
        self,
        keys: Optional[Union[List[str], SerializeDict]] = None,
        output_dir: Optional[str] = None,
    ):
        r"""Save the resources specified by :attr:`keys` in binary format.

        Args:
            keys: list or dict:

                - If :attr:`keys` is a list, the objects corresponding to those keys
                  are saved
                - If :attr:`keys` is a dict mapping from a key to a serialize
                  function, then the serialize function will be used to save
                  the object corresponding to that key
                - If :attr:`keys` is None, all objects in this resource will be
                  saved.
            output_dir:
                A directory specifying the location to save the resources.
        """

        # TODO: use a default save directory like default_save_dir() if None
        if output_dir is None:
            output_dir = "./"

        if keys is None:
            keys = list(self._resources.keys())

        # pylint: disable=isinstance-second-argument-not-valid-type
        # TODO: disable until fix: https://github.com/PyCQA/pylint/issues/3507
        if isinstance(keys, List):
            for key in keys:
                with open(Path(output_dir, f"{key}.pkl"), "wb") as f:
                    pickle.dump(
                        self._resources.get(key), f, pickle.HIGHEST_PROTOCOL
                    )
        else:
            for key, serializer in keys.items():
                serializer(self._resources[key], Path(output_dir, f"{key}.pkl"))

    def keys(self) -> KeysView:
        r"""Return all keys of the resources."""
        return self._resources.keys()

    def contains(self, key: str) -> bool:
        """Return whether the specified key exists."""
        return key in self._resources.keys()

    def get(self, key: str):
        r"""Get the corresponding resource by specifying the key."""
        return self._resources.get(key)

    def update(self, **kwargs):
        r"""Update the resources."""
        self._resources.update(**kwargs)

    def remove(self, key: str):
        r"""Remove the corresponding resource by specifying the key."""
        del self._resources[key]

    def load(
        self,
        keys: Union[List[str], DeserializeDict],
        path: Optional[str] = None,
    ):
        r"""Load the resources specified by :attr:`keys`.

        Args:
            keys list or dict:

                - If :attr:`keys` is a list, the objects corresponding to those keys
                  are loaded
                - If :attr:`keys` is a dict mapping from a key to a deserialize
                  function, then the deserialize function will be used to load
                  the object corresponding to that key
            path: str
                A directory specifying the location to load the resources from.
        """

        # TODO: use a default save directory like default_save_dir() if None
        if path is None:
            path = "./"

        # pylint: disable=isinstance-second-argument-not-valid-type
        # TODO: disable until fix: https://github.com/PyCQA/pylint/issues/3507
        if isinstance(keys, List):
            for key in keys:
                with open(Path(path, f"{key}.pkl"), "rb") as f:
                    self._resources[key] = pickle.load(f)
        else:
            for key, deserializer in keys.items():
                self._resources[key] = deserializer(Path(path, f"{key}.pkl"))
