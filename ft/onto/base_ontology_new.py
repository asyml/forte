# ***automatically_generated***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""


Automatically generated ontology base_ontology_new. Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Group
from forte.data.ontology.top import Link
from typing import Optional
from typing import Set


__all__ = [
    "Token",
    "Document",
    "SpecialDocument",
    "Sentence",
    "Utterance",
    "PredicateArgument",
    "EntityMention",
    "PredicateMention",
    "PredicateLink",
    "Dependency",
    "EnhancedDependency",
    "RelationLink",
    "CoreferenceGroup",
]


class Token(Annotation):
    """
    A span based annotation :class:`Token`, used to represent a token or a word.

    Attributes:
        pos (Optional[str])
        ud_xpos (Optional[str])	Language specific pos tag. Used in CoNLL-U Format. Refer to https://universaldependencies.org/format.html
        lemma (Optional[str])	Lemma or stem of word form.
        ner (Optional[str])
        sense (Optional[str])
        is_root (Optional[bool])

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.pos: Optional[str] = None
        self.ud_xpos: Optional[str] = None
        self.lemma: Optional[str] = None
        self.ner: Optional[str] = None
        self.sense: Optional[str] = None
        self.is_root: Optional[bool] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['pos'] = self.pos
        state['ud_xpos'] = self.ud_xpos
        state['lemma'] = self.lemma
        state['ner'] = self.ner
        state['sense'] = self.sense
        state['is_root'] = self.is_root
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.pos = state.get('pos', None) 
        self.ud_xpos = state.get('ud_xpos', None) 
        self.lemma = state.get('lemma', None) 
        self.ner = state.get('ner', None) 
        self.sense = state.get('sense', None) 
        self.is_root = state.get('is_root', None) 

    @property
    def pos(self):
        return self.pos

    @pos.setter
    def pos(self, pos: Optional[str]):
        self.set_fields(pos=pos)

    @property
    def ud_xpos(self):
        return self.ud_xpos

    @ud_xpos.setter
    def ud_xpos(self, ud_xpos: Optional[str]):
        self.set_fields(ud_xpos=ud_xpos)

    @property
    def lemma(self):
        return self.lemma

    @lemma.setter
    def lemma(self, lemma: Optional[str]):
        self.set_fields(lemma=lemma)

    @property
    def ner(self):
        return self.ner

    @ner.setter
    def ner(self, ner: Optional[str]):
        self.set_fields(ner=ner)

    @property
    def sense(self):
        return self.sense

    @sense.setter
    def sense(self, sense: Optional[str]):
        self.set_fields(sense=sense)

    @property
    def is_root(self):
        return self.is_root

    @is_root.setter
    def is_root(self, is_root: Optional[bool]):
        self.set_fields(is_root=is_root)


class Document(Annotation):
    """
    A span based annotation `Document`, normally used to represent a document.


    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)

    def __getstate__(self): 
        state = super().__getstate__()
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)


class SpecialDocument(Document):
    """


    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)

    def __getstate__(self): 
        state = super().__getstate__()
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)


class Sentence(Annotation):
    """
    A span based annotation `Sentence`, normally used to represent a sentence.


    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)

    def __getstate__(self): 
        state = super().__getstate__()
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)


class Utterance(Annotation):
    """
    A span based annotation `Utterance`, normally used to represent an utterance in dialogue.


    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)

    def __getstate__(self): 
        state = super().__getstate__()
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)


class PredicateArgument(Annotation):
    """
    A span based annotation `PredicateArgument`, normally used to represent an argument of a predicate, can be linked to the predicate via the predicate link.

    Attributes:
        ner_type (Optional[str])
        predicate_lemma (Optional[str])
        is_verb (Optional[bool])

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.ner_type: Optional[str] = None
        self.predicate_lemma: Optional[str] = None
        self.is_verb: Optional[bool] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['ner_type'] = self.ner_type
        state['predicate_lemma'] = self.predicate_lemma
        state['is_verb'] = self.is_verb
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.ner_type = state.get('ner_type', None) 
        self.predicate_lemma = state.get('predicate_lemma', None) 
        self.is_verb = state.get('is_verb', None) 

    @property
    def ner_type(self):
        return self.ner_type

    @ner_type.setter
    def ner_type(self, ner_type: Optional[str]):
        self.set_fields(ner_type=ner_type)

    @property
    def predicate_lemma(self):
        return self.predicate_lemma

    @predicate_lemma.setter
    def predicate_lemma(self, predicate_lemma: Optional[str]):
        self.set_fields(predicate_lemma=predicate_lemma)

    @property
    def is_verb(self):
        return self.is_verb

    @is_verb.setter
    def is_verb(self, is_verb: Optional[bool]):
        self.set_fields(is_verb=is_verb)


class EntityMention(Annotation):
    """
    A span based annotation `EntityMention`, normally used to represent an Entity Mention in a piece of text.

    Attributes:
        ner_type (Optional[str])

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.ner_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['ner_type'] = self.ner_type
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.ner_type = state.get('ner_type', None) 

    @property
    def ner_type(self):
        return self.ner_type

    @ner_type.setter
    def ner_type(self, ner_type: Optional[str]):
        self.set_fields(ner_type=ner_type)


class PredicateMention(Annotation):
    """
    A span based annotation `PredicateMention`, normally used to represent a predicate (normally verbs) in a piece of text.

    Attributes:
        ner_type (Optional[str])

    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.ner_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['ner_type'] = self.ner_type
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.ner_type = state.get('ner_type', None) 

    @property
    def ner_type(self):
        return self.ner_type

    @ner_type.setter
    def ner_type(self, ner_type: Optional[str]):
        self.set_fields(ner_type=ner_type)


class PredicateLink(Link):
    """
    A `Link` type entry which represent a semantic role link between a predicate and its argument.

    Attributes:
        arg_type (Optional[str])	The predicate link type.

    """

    ParentType = PredicateMention

    ChildType = PredicateArgument

    def __init__(self, pack: DataPack, parent: Optional[DataPack] = None, child: Optional[DataPack] = None):
        super().__init__(pack, parent, child)
        self.arg_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['arg_type'] = self.arg_type
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.arg_type = state.get('arg_type', None) 

    @property
    def arg_type(self):
        return self.arg_type

    @arg_type.setter
    def arg_type(self, arg_type: Optional[str]):
        self.set_fields(arg_type=arg_type)


class Dependency(Link):
    """
    A `Link` type entry which represent a syntactic dependency.

    Attributes:
        dep_label (Optional[str])	The dependency label.

    """

    ParentType = Token

    ChildType = Token

    def __init__(self, pack: DataPack, parent: Optional[DataPack] = None, child: Optional[DataPack] = None):
        super().__init__(pack, parent, child)
        self.dep_label: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['dep_label'] = self.dep_label
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.dep_label = state.get('dep_label', None) 

    @property
    def dep_label(self):
        return self.dep_label

    @dep_label.setter
    def dep_label(self, dep_label: Optional[str]):
        self.set_fields(dep_label=dep_label)


class EnhancedDependency(Link):
    """
    A `Link` type entry which represent a enhanced dependency: 
     https://universaldependencies.org/u/overview/enhanced-syntax.html

    Attributes:
        dep_label (Optional[str])	The enhanced dependency label in Universal Dependency.

    """

    ParentType = Token

    ChildType = Token

    def __init__(self, pack: DataPack, parent: Optional[DataPack] = None, child: Optional[DataPack] = None):
        super().__init__(pack, parent, child)
        self.dep_label: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['dep_label'] = self.dep_label
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.dep_label = state.get('dep_label', None) 

    @property
    def dep_label(self):
        return self.dep_label

    @dep_label.setter
    def dep_label(self, dep_label: Optional[str]):
        self.set_fields(dep_label=dep_label)


class RelationLink(Link):
    """
    A `Link` type entry which represent a relation.

    Attributes:
        rel_type (Optional[str])	The type of the relation.

    """

    ParentType = EntityMention

    ChildType = EntityMention

    def __init__(self, pack: DataPack, parent: Optional[DataPack] = None, child: Optional[DataPack] = None):
        super().__init__(pack, parent, child)
        self.rel_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['rel_type'] = self.rel_type
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
        self.rel_type = state.get('rel_type', None) 

    @property
    def rel_type(self):
        return self.rel_type

    @rel_type.setter
    def rel_type(self, rel_type: Optional[str]):
        self.set_fields(rel_type=rel_type)


class CoreferenceGroup(Group):
    """
    A group type entry that take `EntityMention`, as members, used to represent coreferent group of entities.


    """

    def __init__(self, pack: DataPack, members: Optional[Set[DataPack]] = None):
        super().__init__(pack, members)

    def __getstate__(self): 
        state = super().__getstate__()
        return state

    def __setstate__(self, state): 
        state = super().__setstate__(state)
