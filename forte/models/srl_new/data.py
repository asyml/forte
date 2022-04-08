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

from typing import List, NamedTuple, Tuple

import numpy as np
from mypy_extensions import TypedDict


class SRLSpan(NamedTuple):
    predicate: int
    start: int
    end: int
    label: str


class Span(NamedTuple):
    start: int
    end: int
    label: int


class RawExample(TypedDict):
    speakers: List[List[str]]
    doc_key: str
    sentences: List[List[str]]
    srl: List[List[Tuple[int, int, int, str]]]
    constituents: List[List[Span]]
    cluster: List
    ner: List[List[Span]]


class Example(TypedDict):
    text: List[str]
    text_ids: np.ndarray
    srl: List[SRLSpan]
