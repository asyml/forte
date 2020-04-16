# ***automatically_generated***
# ***source json:forte/ontology_specs/race_qa_onto.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology race_multi_choice_qa_ontology. Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from ft.onto.base_ontology import Document
from typing import List
from typing import Optional


__all__ = [
    "RaceDocument",
    "Passage",
    "Option",
    "Question",
]


class RaceDocument(Document):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class Passage(Document):
    """
    Attributes:
        _passage_id (Optional[str])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._passage_id: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['passage_id'] = state.pop('_passage_id')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._passage_id = state.get('passage_id', None) 

    @property
    def passage_id(self):
        return self._passage_id

    @passage_id.setter
    def passage_id(self, passage_id: Optional[str]):
        self.set_fields(_passage_id=passage_id)


class Option(Annotation):
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class Question(Annotation):
    """
    Attributes:
        _options (Optional[List[int]])
        _answers (Optional[List[int]])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._options: Optional[List[int]] = []
        self._answers: Optional[List[int]] = []

    def __getstate__(self): 
        state = super().__getstate__()
        state['options'] = state.pop('_options')
        state['answers'] = state.pop('_answers')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._options = state.get('options', None) 
        self._answers = state.get('answers', None) 

    @property
    def options(self):
        return [self.pack.get_entry(tid) for tid in self._options]

    @options.setter
    def options(self, options: Optional[List[Option]]):
        options = [] if options is None else options
        self.set_fields(_options=[obj.tid for obj in options])

    def num_options(self):
        return len(self._options)

    def clear_options(self):
        [self.pack.delete_entry(self.pack.get_entry(tid)) for tid in self._options]
        self._options.clear()

    def add_options(self, a_options: Option):
        self._options.append(a_options.tid)

    @property
    def answers(self):
        return self._answers

    @answers.setter
    def answers(self, answers: Optional[List[int]]):
        answers = [] if answers is None else answers
        self.set_fields(_answers=answers)

    def num_answers(self):
        return len(self._answers)

    def clear_answers(self):
        self._answers.clear()

    def add_answers(self, a_answers: int):
        self._answers.append(a_answers)
