# ***automatically_generated***
# ***source json:../../../../../../Documents/forte_develop/forte/forte/ontology_specs/base_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology base_ontology. Do not change manually.
"""

from dataclasses import dataclass
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology.core import Entry
from forte.data.ontology.core import FDict
from forte.data.ontology.core import FList
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Generics
from forte.data.ontology.top import Group
from forte.data.ontology.top import Link
from forte.data.ontology.top import MultiPackLink
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

__all__ = [
    "Token",
    "Subword",
    "Classification",
    "Document",
    "Sentence",
    "Phrase",
    "UtteranceContext",
    "Utterance",
    "PredicateArgument",
    "EntityMention",
    "EventMention",
    "PredicateMention",
    "PredicateLink",
    "Dependency",
    "EnhancedDependency",
    "RelationLink",
    "CrossDocEntityRelation",
    "CoreferenceGroup",
    "EventRelation",
    "CrossDocEventRelation",
    "ConstituentNode",
    "Title",
]


@dataclass
class Token(Annotation):
    """
    A span based annotation :class:`Token`, used to represent a token or a word.
    Attributes:
        pos (Optional[str]):
        ud_xpos (Optional[str]):	Language specific pos tag. Used in CoNLL-U Format. Refer to https://universaldependencies.org/format.html
        lemma (Optional[str]):	Lemma or stem of word form.
        chunk (Optional[str]):
        ner (Optional[str]):
        sense (Optional[str]):
        is_root (Optional[bool]):
        ud_features (Dict[str, str]):
        ud_misc (Dict[str, str]):
    """

    pos: Optional[str]
    ud_xpos: Optional[str]
    lemma: Optional[str]
    chunk: Optional[str]
    ner: Optional[str]
    sense: Optional[str]
    is_root: Optional[bool]
    ud_features: Dict[str, str]
    ud_misc: Dict[str, str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.pos: Optional[str] = None
        self.ud_xpos: Optional[str] = None
        self.lemma: Optional[str] = None
        self.chunk: Optional[str] = None
        self.ner: Optional[str] = None
        self.sense: Optional[str] = None
        self.is_root: Optional[bool] = None
        self.ud_features: Dict[str, str] = dict()
        self.ud_misc: Dict[str, str] = dict()


@dataclass
class Subword(Annotation):
    """
    Used to represent subword tokenization results.
    Attributes:
        is_first_segment (Optional[bool]):
    """

    is_first_segment: Optional[bool]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.is_first_segment: Optional[bool] = None


@dataclass
class Classification(Generics):
    """
    Used to store values for classification prediction
    Attributes:
        classification_result (Dict[str, float]):
    """

    classification_result: Dict[str, float]

    def __init__(self, pack: DataPack):
        super().__init__(pack)
        self.classification_result: Dict[str, float] = dict()


@dataclass
class Document(Annotation):
    """
    A span based annotation `Document`, normally used to represent a document.
    Attributes:
        document_class (List[str]):	A list of class names that the document belongs to.
        sentiment (Dict[str, float]):
        classifications (FDict[str, Classification]):	Stores the classification results for this document. The key is the name/task of the classification, the value is an classification object storing the results.
    """

    document_class: List[str]
    sentiment: Dict[str, float]
    classifications: FDict[str, Classification]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.document_class: List[str] = []
        self.sentiment: Dict[str, float] = dict()
        self.classifications: FDict[str, Classification] = FDict(self)


@dataclass
class Sentence(Annotation):
    """
    A span based annotation `Sentence`, normally used to represent a sentence.
    Attributes:
        speaker (Optional[str]):
        part_id (Optional[int]):
        sentiment (Dict[str, float]):
        classification (Dict[str, float]):
        classifications (FDict[str, Classification]):	Stores the classification results for this sentence. The key is the name/task of the classification, the value is an classification object storing the results.
    """

    speaker: Optional[str]
    part_id: Optional[int]
    sentiment: Dict[str, float]
    classification: Dict[str, float]
    classifications: FDict[str, Classification]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.speaker: Optional[str] = None
        self.part_id: Optional[int] = None
        self.sentiment: Dict[str, float] = dict()
        self.classification: Dict[str, float] = dict()
        self.classifications: FDict[str, Classification] = FDict(self)


@dataclass
class Phrase(Annotation):
    """
    A span based annotation `Phrase`.
    Attributes:
        phrase_type (Optional[str]):
        headword (Optional[Token]):
    """

    phrase_type: Optional[str]
    headword: Optional[Token]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.phrase_type: Optional[str] = None
        self.headword: Optional[Token] = None


@dataclass
class UtteranceContext(Annotation):
    """
    `UtteranceContext` represents the context part in dialogue.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


@dataclass
class Utterance(Annotation):
    """
    A span based annotation `Utterance`, normally used to represent an utterance in dialogue.
    Attributes:
        speaker (Optional[str]):
    """

    speaker: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.speaker: Optional[str] = None


@dataclass
class PredicateArgument(Annotation):
    """
    A span based annotation `PredicateArgument`, normally used to represent an argument of a predicate, can be linked to the predicate via the predicate link.
    Attributes:
        ner_type (Optional[str]):
        predicate_lemma (Optional[str]):
        is_verb (Optional[bool]):
    """

    ner_type: Optional[str]
    predicate_lemma: Optional[str]
    is_verb: Optional[bool]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.ner_type: Optional[str] = None
        self.predicate_lemma: Optional[str] = None
        self.is_verb: Optional[bool] = None


@dataclass
class EntityMention(Annotation):
    """
    A span based annotation `EntityMention`, normally used to represent an Entity Mention in a piece of text.
    Attributes:
        ner_type (Optional[str]):
    """

    ner_type: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.ner_type: Optional[str] = None


@dataclass
class EventMention(Annotation):
    """
    A span based annotation `EventMention`, used to refer to a mention of an event.
    Attributes:
        event_type (Optional[str]):
    """

    event_type: Optional[str]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.event_type: Optional[str] = None


@dataclass
class PredicateMention(Phrase):
    """
    A span based annotation `PredicateMention`, normally used to represent a predicate (normally verbs) in a piece of text.
    Attributes:
        predicate_lemma (Optional[str]):
        framenet_id (Optional[str]):
        is_verb (Optional[bool]):
    """

    predicate_lemma: Optional[str]
    framenet_id: Optional[str]
    is_verb: Optional[bool]

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.predicate_lemma: Optional[str] = None
        self.framenet_id: Optional[str] = None
        self.is_verb: Optional[bool] = None


@dataclass
class PredicateLink(Link):
    """
    A `Link` type entry which represent a semantic role link between a predicate and its argument.
    Attributes:
        arg_type (Optional[str]):	The predicate link type.
    """

    arg_type: Optional[str]

    ParentType = PredicateMention
    ChildType = PredicateArgument

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.arg_type: Optional[str] = None


@dataclass
class Dependency(Link):
    """
    A `Link` type entry which represent a syntactic dependency.
    Attributes:
        dep_label (Optional[str]):	The dependency label.
        rel_type (Optional[str]):
    """

    dep_label: Optional[str]
    rel_type: Optional[str]

    ParentType = Token
    ChildType = Token

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.dep_label: Optional[str] = None
        self.rel_type: Optional[str] = None


@dataclass
class EnhancedDependency(Link):
    """
    A `Link` type entry which represent a enhanced dependency: 
     https://universaldependencies.org/u/overview/enhanced-syntax.html
    Attributes:
        dep_label (Optional[str]):	The enhanced dependency label in Universal Dependency.
    """

    dep_label: Optional[str]

    ParentType = Token
    ChildType = Token

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.dep_label: Optional[str] = None


@dataclass
class RelationLink(Link):
    """
    A `Link` type entry which represent a relation between two entity mentions
    Attributes:
        rel_type (Optional[str]):	The type of the relation.
    """

    rel_type: Optional[str]

    ParentType = EntityMention
    ChildType = EntityMention

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.rel_type: Optional[str] = None


@dataclass
class CrossDocEntityRelation(MultiPackLink):
    """
    A `Link` type entry which represent a relation between two entity mentions across the packs.
    Attributes:
        rel_type (Optional[str]):	The type of the relation.
    """

    rel_type: Optional[str]

    ParentType = EntityMention
    ChildType = EntityMention

    def __init__(self, pack: MultiPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.rel_type: Optional[str] = None


@dataclass
class CoreferenceGroup(Group):
    """
    A group type entry that take `EntityMention`, as members, used to represent coreferent group of entities.
    """

    MemberType = EntityMention

    def __init__(self, pack: DataPack, members: Optional[Iterable[Entry]] = None):
        super().__init__(pack, members)


@dataclass
class EventRelation(Link):
    """
    A `Link` type entry which represent a relation between two event mentions.
    Attributes:
        rel_type (Optional[str]):	The type of the relation.
    """

    rel_type: Optional[str]

    ParentType = EventMention
    ChildType = EventMention

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.rel_type: Optional[str] = None


@dataclass
class CrossDocEventRelation(MultiPackLink):
    """
    A `Link` type entry which represent a relation between two event mentions across the packs.
    Attributes:
        rel_type (Optional[str]):	The type of the relation.
    """

    rel_type: Optional[str]

    ParentType = EventMention
    ChildType = EventMention

    def __init__(self, pack: MultiPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self.rel_type: Optional[str] = None


@dataclass
class ConstituentNode(Annotation):
    """
    A span based annotation `ConstituentNode` to represent constituents in constituency parsing. This can also sentiment values annotated on the nodes.
    Attributes:
        label (Optional[str]):
        sentiment (Dict[str, float]):
        is_root (Optional[bool]):
        is_leaf (Optional[bool]):
        parent_node (Optional['ConstituentNode']):
        children_nodes (FList['ConstituentNode']):
    """

    label: Optional[str]
    sentiment: Dict[str, float]
    is_root: Optional[bool]
    is_leaf: Optional[bool]
    parent_node: Optional['ConstituentNode']
    children_nodes: FList['ConstituentNode']

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.label: Optional[str] = None
        self.sentiment: Dict[str, float] = dict()
        self.is_root: Optional[bool] = None
        self.is_leaf: Optional[bool] = None
        self.parent_node: Optional['ConstituentNode'] = None
        self.children_nodes: FList['ConstituentNode'] = FList(self)


@dataclass
class Title(Annotation):
    """
    A span based annotation `Title`, normally used to represent a title.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
