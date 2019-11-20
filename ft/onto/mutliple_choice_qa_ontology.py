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

from typing import List

from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from ft.onto.base_ontology import Document


class Passage(Document):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._passage_id: str

    def set_passage_id(self, pid: str):
        self.set_fields(_passage_id=pid)

    @property
    def passage_id(self):
        return self._passage_id


# pylint: disable=useless-super-delegation
class Option(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class Question(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._options: List[Option]
        self._answers: List[int]

    def set_options(self, options: List[Option]):
        self.set_fields(_options=options)

    def set_answers(self, answers: List[int]):
        self.set_fields(_answers=answers)

    @property
    def options(self):
        return self._options

    @property
    def num_options(self):
        return len(self._options)

    @property
    def answers(self):
        return self._answers


# pylint: disable=useless-super-delegation
class Article(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
