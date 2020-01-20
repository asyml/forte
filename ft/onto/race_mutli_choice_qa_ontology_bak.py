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

from typing import List, Any

from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from ft.onto.base_ontology import Document


class Passage(Document):
    def __init__(self, pack: DataPack, begin: int, end: int) -> None:
        super().__init__(pack, begin, end)
        self._passage_id: str

    def set_passage_id(self, pid: str) -> None:
        self.set_fields(_passage_id=pid)

    @property
    def passage_id(self):
        return self._passage_id


# pylint: disable=useless-super-delegation
class Option(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class Question(Annotation):
    OptionType: Any = Option

    def __init__(self, pack: DataPack, begin: int, end: int) -> None:
        super().__init__(pack, begin, end)
        self._options: List[int] = []
        self._answers: List[int] = []

    def get_options(self) -> List[Option]:
        return [self.pack.get_entry(tid) for tid in self._options]

    def set_options(self, options: List[Option]) -> None:
        options_tid = []
        for option in options:
            if not isinstance(option, self.OptionType):
                raise TypeError(
                    f"The option of {type(self)} should be an "
                    f"instance of {self.OptionType}, but got {type(option)}")
            options_tid.append(option.tid)
        self.set_fields(_options=options_tid)

    def clear_options(self) -> None:
        self.set_options([])

    @property
    def options(self) -> List[int]:
        return self._options

    @property
    def num_options(self) -> int:
        return len(self._options)

    def get_answers(self) -> List[int]:
        return self._answers

    def set_answers(self, answers: List[int]) -> None:
        self.set_fields(_answers=answers)

    def clear_answers(self) -> None:
        self.set_answers([])

    @property
    def answers(self) -> List[int]:
        return self._answers

    @property
    def num_answers(self) -> int:
        return len(self._answers)


# pylint: disable=useless-super-delegation
class Article(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
