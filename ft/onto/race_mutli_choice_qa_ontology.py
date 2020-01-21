# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""


Automatically generated file. Do not change manually.
"""
import forte.data.data_pack
import forte.data.ontology.top
import ft.onto
import ft.onto.base_ontology
import typing


__all__ = []


__all__.extend('Passage')


class Passage(ft.onto.base_ontology.Document):
    """

    Attributes:
        passage_id (typing.Optional[str])

    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.passage_id: typing.Optional[str] = None

    @property
    def passage_id(self):
        return self.passage_id

    @passage_id.setter
    def passage_id(self, passage_id: typing.Optional[str]):
        self.set_fields(passage_id=passage_id)


__all__.extend('Option')


class Option(forte.data.ontology.top.Annotation):
    """


    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)


__all__.extend('Question')


class Question(forte.data.ontology.top.Annotation):
    """

    Attributes:
        options (typing.Optional[typing.List[ft.onto.race_mutli_choice_qa_ontology.Option]])
        answers (typing.Optional[typing.List[int]])

    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.options: typing.Optional[typing.List[ft.onto.race_mutli_choice_qa_ontology.Option]] = None
        self.answers: typing.Optional[typing.List[int]] = None

    @property
    def options(self):
        return self.options

    @options.setter
    def options(self, options: typing.Optional[typing.List[ft.onto.race_mutli_choice_qa_ontology.Option]]):
        self.set_fields(options=[item.tid for item in options])

    def num_options(self):
        return len(self.options)

    def clear_options(self):
        self.options.clear()

    def add_options(self, a_options: ft.onto.race_mutli_choice_qa_ontology.Option):
        self.options.append(a_options)

    @property
    def answers(self):
        return self.answers

    @answers.setter
    def answers(self, answers: typing.Optional[typing.List[int]]):
        self.set_fields(answers=[item for item in answers])

    def num_answers(self):
        return len(self.answers)

    def clear_answers(self):
        self.answers.clear()

    def add_answers(self, a_answers: int):
        self.answers.append(a_answers)
