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
import typing


__all__ = []


__all__.extend('Token')


class Token(forte.data.ontology.top.Annotation):
    """
    A span based annotation :class:`Token`, used to represent a token or a word.

    Attributes:
        pos (typing.Optional[str])
        ud_xpos (typing.Optional[str])	Language specific pos tag. Used in CoNLL-U Format. Refer to https://universaldependencies.org/format.html
        lemma (typing.Optional[str])	Lemma or stem of word form.
        ner (typing.Optional[str])
        sense (typing.Optional[str])
        is_root (typing.Optional[bool])

    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.pos: typing.Optional[str] = None
        self.ud_xpos: typing.Optional[str] = None
        self.lemma: typing.Optional[str] = None
        self.ner: typing.Optional[str] = None
        self.sense: typing.Optional[str] = None
        self.is_root: typing.Optional[bool] = None

    @property
    def pos(self):
        return self.pos

    @pos.setter
    def pos(self, pos: typing.Optional[str]):
        self.set_fields(pos=pos)

    @property
    def ud_xpos(self):
        return self.ud_xpos

    @ud_xpos.setter
    def ud_xpos(self, ud_xpos: typing.Optional[str]):
        self.set_fields(ud_xpos=ud_xpos)

    @property
    def lemma(self):
        return self.lemma

    @lemma.setter
    def lemma(self, lemma: typing.Optional[str]):
        self.set_fields(lemma=lemma)

    @property
    def ner(self):
        return self.ner

    @ner.setter
    def ner(self, ner: typing.Optional[str]):
        self.set_fields(ner=ner)

    @property
    def sense(self):
        return self.sense

    @sense.setter
    def sense(self, sense: typing.Optional[str]):
        self.set_fields(sense=sense)

    @property
    def is_root(self):
        return self.is_root

    @is_root.setter
    def is_root(self, is_root: typing.Optional[bool]):
        self.set_fields(is_root=is_root)


__all__.extend('Document')


class Document(forte.data.ontology.top.Annotation):
    """
    A span based annotation `Document`, normally used to represent a document.


    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)


__all__.extend('Sentence')


class Sentence(forte.data.ontology.top.Annotation):
    """
    A span based annotation `Sentence`, normally used to represent a sentence.


    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)


__all__.extend('Utterance')


class Utterance(forte.data.ontology.top.Annotation):
    """
    A span based annotation `Utterance`, normally used to represent an utterance in dialogue.


    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)


__all__.extend('PredicateArgument')


class PredicateArgument(forte.data.ontology.top.Annotation):
    """
    A span based annotation `PredicateArgument`, normally used to represent an argument of a predicate, can be linked to the predicate via the predicate link.

    Attributes:
        ner_type (typing.Optional[str])

    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.ner_type: typing.Optional[str] = None

    @property
    def ner_type(self):
        return self.ner_type

    @ner_type.setter
    def ner_type(self, ner_type: typing.Optional[str]):
        self.set_fields(ner_type=ner_type)


__all__.extend('EntityMention')


class EntityMention(forte.data.ontology.top.Annotation):
    """
    A span based annotation `EntityMention`, normally used to represent an Entity Mention in a piece of text.

    Attributes:
        ner_type (typing.Optional[str])

    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.ner_type: typing.Optional[str] = None

    @property
    def ner_type(self):
        return self.ner_type

    @ner_type.setter
    def ner_type(self, ner_type: typing.Optional[str]):
        self.set_fields(ner_type=ner_type)


__all__.extend('PredicateMention')


class PredicateMention(forte.data.ontology.top.Annotation):
    """
    A span based annotation `PredicateMention`, normally used to represent a predicate (normally verbs) in a piece of text.

    Attributes:
        ner_type (typing.Optional[str])

    """

    def __init__(self, pack: forte.data.base_pack.PackType, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.ner_type: typing.Optional[str] = None

    @property
    def ner_type(self):
        return self.ner_type

    @ner_type.setter
    def ner_type(self, ner_type: typing.Optional[str]):
        self.set_fields(ner_type=ner_type)


__all__.extend('PredicateLink')


class PredicateLink(forte.data.ontology.top.Link):
    """
    A `Link` type entry which represent a semantic role link between a predicate and its argument.

    Attributes:
        arg_type (typing.Optional[str])	The predicate link type.

    """
    parent_type = ft.onto.base_ontology.PredicateMention
    child_type = ft.onto.base_ontology.PredicateArgument

    def __init__(self, pack: forte.data.base_pack.PackType, parent: typing.Optional[forte.data.ontology.core.Entry] = None, child: typing.Optional[forte.data.ontology.core.Entry] = None):
        super().__init__(pack, parent, child)
        self.arg_type: typing.Optional[str] = None

    @property
    def arg_type(self):
        return self.arg_type

    @arg_type.setter
    def arg_type(self, arg_type: typing.Optional[str]):
        self.set_fields(arg_type=arg_type)


__all__.extend('Dependency')


class Dependency(forte.data.ontology.top.Link):
    """
    A `Link` type entry which represent a syntactic dependency.

    Attributes:
        dep_label (typing.Optional[str])	The dependency label.

    """
    parent_type = ft.onto.base_ontology.Token
    child_type = ft.onto.base_ontology.Token

    def __init__(self, pack: forte.data.base_pack.PackType, parent: typing.Optional[forte.data.ontology.core.Entry] = None, child: typing.Optional[forte.data.ontology.core.Entry] = None):
        super().__init__(pack, parent, child)
        self.dep_label: typing.Optional[str] = None

    @property
    def dep_label(self):
        return self.dep_label

    @dep_label.setter
    def dep_label(self, dep_label: typing.Optional[str]):
        self.set_fields(dep_label=dep_label)


__all__.extend('EnhancedDependency')


class EnhancedDependency(forte.data.ontology.top.Link):
    """
    A `Link` type entry which represent a enhanced dependency: 
     https://universaldependencies.org/u/overview/enhanced-syntax.html

    Attributes:
        dep_label (typing.Optional[str])	The enhanced dependency label in Universal Dependency.

    """
    parent_type = ft.onto.base_ontology.Token
    child_type = ft.onto.base_ontology.Token

    def __init__(self, pack: forte.data.base_pack.PackType, parent: typing.Optional[forte.data.ontology.core.Entry] = None, child: typing.Optional[forte.data.ontology.core.Entry] = None):
        super().__init__(pack, parent, child)
        self.dep_label: typing.Optional[str] = None

    @property
    def dep_label(self):
        return self.dep_label

    @dep_label.setter
    def dep_label(self, dep_label: typing.Optional[str]):
        self.set_fields(dep_label=dep_label)


__all__.extend('RelationLink')


class RelationLink(forte.data.ontology.top.Link):
    """
    A `Link` type entry which represent a relation.

    Attributes:
        rel_type (typing.Optional[str])	The type of the relation.

    """
    parent_type = ft.onto.base_ontology.EntityMention
    child_type = ft.onto.base_ontology.EntityMention

    def __init__(self, pack: forte.data.base_pack.PackType, parent: typing.Optional[forte.data.ontology.core.Entry] = None, child: typing.Optional[forte.data.ontology.core.Entry] = None):
        super().__init__(pack, parent, child)
        self.rel_type: typing.Optional[str] = None

    @property
    def rel_type(self):
        return self.rel_type

    @rel_type.setter
    def rel_type(self, rel_type: typing.Optional[str]):
        self.set_fields(rel_type=rel_type)


__all__.extend('CoreferenceGroup')


class CoreferenceGroup(forte.data.ontology.top.Group):
    """
    A group type entry that take `EntityMention`, as members, used to represent coreferent group of entities.


    """
    member_type = ft.onto.base_ontology.EntityMention

    def __init__(self, pack: forte.data.container.EntryContainer, members: typing.Optional[typing.Set[forte.data.ontology.core.Entry]] = None):
        super().__init__(pack, members)
