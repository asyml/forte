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
from abc import ABC
from typing import Union
import numpy as np

from forte.processors.base.pack_processor import PackProcessor

__all__ = [
    "QueryProcessor"
]

QueryType = Union[str, np.ndarray]


class QueryProcessor(PackProcessor, ABC):
    r"""A base class for all processors that handle query creation for
    information retrieval."""

    def _build_query(self, text: str) -> QueryType:
        r"""Query-related processors need to implement this class to create a
        query from a data pack.

        """
        raise NotImplementedError
