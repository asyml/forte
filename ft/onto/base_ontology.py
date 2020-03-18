# ***automatically_generated***
# ***source json:forte/ontology_specs/base_ontology.json***
# flake8: noqa
# mypy: ignore-errors
# pylint: skip-file
"""
Automatically generated ontology base_ontology. Do not change manually.
"""

from forte.data.data_pack import DataPack
from forte.data.ontology.core import Entry
from forte.data.ontology.top import Annotation
from forte.data.ontology.top import Group
from forte.data.ontology.top import Link
from typing import Dict
from typing import Optional
from typing import Set


__all__ = [
    "Token",
    "Document",
    "Sentence",
    "Phrase",
    "Utterance",
    "PredicateArgument",
    "EntityMention",
    "EventMention",
    "PredicateMention",
    "PredicateLink",
    "Dependency",
    "EnhancedDependency",
    "RelationLink",
    "CoreferenceGroup",
    "EventRelation",
]


class Token(Annotation):
    """
    A span based annotation :class:`Token`, used to represent a token or a word.
    Attributes:
        _pos (Optional[str])
        _ud_xpos (Optional[str])	Language specific pos tag. Used in CoNLL-U Format. Refer to https://universaldependencies.org/format.html
        _lemma (Optional[str])	Lemma or stem of word form.
        _chunk (Optional[str])
        _ner (Optional[str])
        _sense (Optional[str])
        _is_root (Optional[bool])
        _ud_features (Optional[Dict[str, str]])
        _ud_misc (Optional[Dict[str, str]])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._pos: Optional[str] = None
        self._ud_xpos: Optional[str] = None
        self._lemma: Optional[str] = None
        self._chunk: Optional[str] = None
        self._ner: Optional[str] = None
        self._sense: Optional[str] = None
        self._is_root: Optional[bool] = None
        self._ud_features: Optional[Dict[str, str]] = {}
        self._ud_misc: Optional[Dict[str, str]] = {}

    def __getstate__(self): 
        state = super().__getstate__()
        state['pos'] = state.pop('_pos')
        state['ud_xpos'] = state.pop('_ud_xpos')
        state['lemma'] = state.pop('_lemma')
        state['chunk'] = state.pop('_chunk')
        state['ner'] = state.pop('_ner')
        state['sense'] = state.pop('_sense')
        state['is_root'] = state.pop('_is_root')
        state['ud_features'] = state.pop('_ud_features')
        state['ud_misc'] = state.pop('_ud_misc')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._pos = state.get('pos', None) 
        self._ud_xpos = state.get('ud_xpos', None) 
        self._lemma = state.get('lemma', None) 
        self._chunk = state.get('chunk', None) 
        self._ner = state.get('ner', None) 
        self._sense = state.get('sense', None) 
        self._is_root = state.get('is_root', None) 
        self._ud_features = state.get('ud_features', None) 
        self._ud_misc = state.get('ud_misc', None) 

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos: Optional[str]):
        self.set_fields(_pos=pos)

    @property
    def ud_xpos(self):
        return self._ud_xpos

    @ud_xpos.setter
    def ud_xpos(self, ud_xpos: Optional[str]):
        self.set_fields(_ud_xpos=ud_xpos)

    @property
    def lemma(self):
        return self._lemma

    @lemma.setter
    def lemma(self, lemma: Optional[str]):
        self.set_fields(_lemma=lemma)

    @property
    def chunk(self):
        return self._chunk

    @chunk.setter
    def chunk(self, chunk: Optional[str]):
        self.set_fields(_chunk=chunk)

    @property
    def ner(self):
        return self._ner

    @ner.setter
    def ner(self, ner: Optional[str]):
        self.set_fields(_ner=ner)

    @property
    def sense(self):
        return self._sense

    @sense.setter
    def sense(self, sense: Optional[str]):
        self.set_fields(_sense=sense)

    @property
    def is_root(self):
        return self._is_root

    @is_root.setter
    def is_root(self, is_root: Optional[bool]):
        self.set_fields(_is_root=is_root)

    @property
    def ud_features(self):
        return self._ud_features

    @ud_features.setter
    def ud_features(self, ud_features: Optional[Dict[str, str]]):
        ud_features = {} if ud_features is None else ud_features
        self.set_fields(_ud_features=ud_features)

    def num_ud_features(self):
        return len(self._ud_features)

    def clear_ud_features(self):
        self._ud_features.clear()

    def add_ud_features(self, key: str, value: str):
        self.ud_features[key] = value

    @property
    def ud_misc(self):
        return self._ud_misc

    @ud_misc.setter
    def ud_misc(self, ud_misc: Optional[Dict[str, str]]):
        ud_misc = {} if ud_misc is None else ud_misc
        self.set_fields(_ud_misc=ud_misc)

    def num_ud_misc(self):
        return len(self._ud_misc)

    def clear_ud_misc(self):
        self._ud_misc.clear()

    def add_ud_misc(self, key: str, value: str):
        self.ud_misc[key] = value


class Document(Annotation):
    """
    A span based annotation `Document`, normally used to represent a document.
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class Sentence(Annotation):
    """
    A span based annotation `Sentence`, normally used to represent a sentence.
    Attributes:
        _speaker (Optional[str])
        _part_id (Optional[int])
        _sentiment (Optional[Dict[str, float]])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._speaker: Optional[str] = None
        self._part_id: Optional[int] = None
        self._sentiment: Optional[Dict[str, float]] = {}

    def __getstate__(self): 
        state = super().__getstate__()
        state['speaker'] = state.pop('_speaker')
        state['part_id'] = state.pop('_part_id')
        state['sentiment'] = state.pop('_sentiment')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._speaker = state.get('speaker', None) 
        self._part_id = state.get('part_id', None) 
        self._sentiment = state.get('sentiment', None) 

    @property
    def speaker(self):
        return self._speaker

    @speaker.setter
    def speaker(self, speaker: Optional[str]):
        self.set_fields(_speaker=speaker)

    @property
    def part_id(self):
        return self._part_id

    @part_id.setter
    def part_id(self, part_id: Optional[int]):
        self.set_fields(_part_id=part_id)

    @property
    def sentiment(self):
        return self._sentiment

    @sentiment.setter
    def sentiment(self, sentiment: Optional[Dict[str, float]]):
        sentiment = {} if sentiment is None else sentiment
        self.set_fields(_sentiment=sentiment)

    def num_sentiment(self):
        return len(self._sentiment)

    def clear_sentiment(self):
        self._sentiment.clear()

    def add_sentiment(self, key: str, value: float):
        self.sentiment[key] = value


class Phrase(Annotation):
    """
    A span based annotation `Phrase`.
    Attributes:
        _phrase_type (Optional[str])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._phrase_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['phrase_type'] = state.pop('_phrase_type')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._phrase_type = state.get('phrase_type', None) 

    @property
    def phrase_type(self):
        return self._phrase_type

    @phrase_type.setter
    def phrase_type(self, phrase_type: Optional[str]):
        self.set_fields(_phrase_type=phrase_type)


class Utterance(Annotation):
    """
    A span based annotation `Utterance`, normally used to represent an utterance in dialogue.
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class PredicateArgument(Annotation):
    """
    A span based annotation `PredicateArgument`, normally used to represent an argument of a predicate, can be linked to the predicate via the predicate link.
    Attributes:
        _ner_type (Optional[str])
        _predicate_lemma (Optional[str])
        _is_verb (Optional[bool])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._ner_type: Optional[str] = None
        self._predicate_lemma: Optional[str] = None
        self._is_verb: Optional[bool] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['ner_type'] = state.pop('_ner_type')
        state['predicate_lemma'] = state.pop('_predicate_lemma')
        state['is_verb'] = state.pop('_is_verb')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._ner_type = state.get('ner_type', None) 
        self._predicate_lemma = state.get('predicate_lemma', None) 
        self._is_verb = state.get('is_verb', None) 

    @property
    def ner_type(self):
        return self._ner_type

    @ner_type.setter
    def ner_type(self, ner_type: Optional[str]):
        self.set_fields(_ner_type=ner_type)

    @property
    def predicate_lemma(self):
        return self._predicate_lemma

    @predicate_lemma.setter
    def predicate_lemma(self, predicate_lemma: Optional[str]):
        self.set_fields(_predicate_lemma=predicate_lemma)

    @property
    def is_verb(self):
        return self._is_verb

    @is_verb.setter
    def is_verb(self, is_verb: Optional[bool]):
        self.set_fields(_is_verb=is_verb)


class EntityMention(Annotation):
    """
    A span based annotation `EntityMention`, normally used to represent an Entity Mention in a piece of text.
    Attributes:
        _ner_type (Optional[str])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._ner_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['ner_type'] = state.pop('_ner_type')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._ner_type = state.get('ner_type', None) 

    @property
    def ner_type(self):
        return self._ner_type

    @ner_type.setter
    def ner_type(self, ner_type: Optional[str]):
        self.set_fields(_ner_type=ner_type)


class EventMention(Annotation):
    """
    A span based annotation `EventMention`, used to refer to a mention of an event.
    Attributes:
        _event_type (Optional[str])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._event_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['event_type'] = state.pop('_event_type')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._event_type = state.get('event_type', None) 

    @property
    def event_type(self):
        return self._event_type

    @event_type.setter
    def event_type(self, event_type: Optional[str]):
        self.set_fields(_event_type=event_type)


class PredicateMention(Annotation):
    """
    A span based annotation `PredicateMention`, normally used to represent a predicate (normally verbs) in a piece of text.
    Attributes:
        _predicate_lemma (Optional[str])
        _framenet_id (Optional[str])
        _is_verb (Optional[bool])
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self._predicate_lemma: Optional[str] = None
        self._framenet_id: Optional[str] = None
        self._is_verb: Optional[bool] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['predicate_lemma'] = state.pop('_predicate_lemma')
        state['framenet_id'] = state.pop('_framenet_id')
        state['is_verb'] = state.pop('_is_verb')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._predicate_lemma = state.get('predicate_lemma', None) 
        self._framenet_id = state.get('framenet_id', None) 
        self._is_verb = state.get('is_verb', None) 

    @property
    def predicate_lemma(self):
        return self._predicate_lemma

    @predicate_lemma.setter
    def predicate_lemma(self, predicate_lemma: Optional[str]):
        self.set_fields(_predicate_lemma=predicate_lemma)

    @property
    def framenet_id(self):
        return self._framenet_id

    @framenet_id.setter
    def framenet_id(self, framenet_id: Optional[str]):
        self.set_fields(_framenet_id=framenet_id)

    @property
    def is_verb(self):
        return self._is_verb

    @is_verb.setter
    def is_verb(self, is_verb: Optional[bool]):
        self.set_fields(_is_verb=is_verb)


class PredicateLink(Link):
    """
    A `Link` type entry which represent a semantic role link between a predicate and its argument.
    Attributes:
        _arg_type (Optional[str])	The predicate link type.
    """
    ParentType = PredicateMention

    ChildType = PredicateArgument

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self._arg_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['arg_type'] = state.pop('_arg_type')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._arg_type = state.get('arg_type', None) 

    @property
    def arg_type(self):
        return self._arg_type

    @arg_type.setter
    def arg_type(self, arg_type: Optional[str]):
        self.set_fields(_arg_type=arg_type)


class Dependency(Link):
    """
    A `Link` type entry which represent a syntactic dependency.
    Attributes:
        _dep_label (Optional[str])	The dependency label.
        _rel_type (Optional[str])
    """
    ParentType = Token

    ChildType = Token

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self._dep_label: Optional[str] = None
        self._rel_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['dep_label'] = state.pop('_dep_label')
        state['rel_type'] = state.pop('_rel_type')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._dep_label = state.get('dep_label', None) 
        self._rel_type = state.get('rel_type', None) 

    @property
    def dep_label(self):
        return self._dep_label

    @dep_label.setter
    def dep_label(self, dep_label: Optional[str]):
        self.set_fields(_dep_label=dep_label)

    @property
    def rel_type(self):
        return self._rel_type

    @rel_type.setter
    def rel_type(self, rel_type: Optional[str]):
        self.set_fields(_rel_type=rel_type)


class EnhancedDependency(Link):
    """
    A `Link` type entry which represent a enhanced dependency: 
     https://universaldependencies.org/u/overview/enhanced-syntax.html
    Attributes:
        _dep_label (Optional[str])	The enhanced dependency label in Universal Dependency.
    """
    ParentType = Token

    ChildType = Token

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self._dep_label: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['dep_label'] = state.pop('_dep_label')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._dep_label = state.get('dep_label', None) 

    @property
    def dep_label(self):
        return self._dep_label

    @dep_label.setter
    def dep_label(self, dep_label: Optional[str]):
        self.set_fields(_dep_label=dep_label)


class RelationLink(Link):
    """
    A `Link` type entry which represent a relation between two entity mentions
    Attributes:
        _rel_type (Optional[str])	The type of the relation.
    """
    ParentType = EntityMention

    ChildType = EntityMention

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self._rel_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['rel_type'] = state.pop('_rel_type')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._rel_type = state.get('rel_type', None) 

    @property
    def rel_type(self):
        return self._rel_type

    @rel_type.setter
    def rel_type(self, rel_type: Optional[str]):
        self.set_fields(_rel_type=rel_type)


class CoreferenceGroup(Group):
    """
    A group type entry that take `EntityMention`, as members, used to represent coreferent group of entities.
    """
    def __init__(self, pack: DataPack, members: Optional[Set[Entry]] = None):
        super().__init__(pack, members)


class EventRelation(Link):
    """
    A `Link` type entry which represent a relation between two event mentions.
    Attributes:
        _rel_type (Optional[str])	The type of the relation.
    """
    ParentType = EventMention

    ChildType = EventMention

    def __init__(self, pack: DataPack, parent: Optional[Entry] = None, child: Optional[Entry] = None):
        super().__init__(pack, parent, child)
        self._rel_type: Optional[str] = None

    def __getstate__(self): 
        state = super().__getstate__()
        state['rel_type'] = state.pop('_rel_type')
        return state

    def __setstate__(self, state): 
        super().__setstate__(state)
        self._rel_type = state.get('rel_type', None) 

    @property
    def rel_type(self):
        return self._rel_type

    @rel_type.setter
    def rel_type(self, rel_type: Optional[str]):
        self.set_fields(_rel_type=rel_type)
