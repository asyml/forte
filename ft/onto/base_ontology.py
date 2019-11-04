"""
This class defines the basic ontology supported by our system
"""
from typing import Optional, Dict, List, Set
from forte.data.data_pack import DataPack
from forte.data.ontology import Entry, Annotation, Link, Group

__all__ = [
    "Token",
    "Sentence",
    "Document",
    "EntityMention",
    "PredicateArgument",
    "PredicateMention",
    "PredicateLink",
    "CoreferenceGroup",
    "Dependency",
    "RelationLink"
]


class Token(Annotation):
    """
    A span based annotation :class:`Token`.

    Args:
        pack (DataPack): The data pack this token belongs to.
        begin (int): The offset of the first character in the token.
        end (int): The offset of the last character in the token + 1.

    Attributes:
        ud_xpos (str): Language specific pos tag. Used in CoNLL-U Format. Refer
        https://universaldependencies.org/format.html
        lemma (str): Lemma or stem of word form.
        is_root (bool): If the token is a root of, say, dependency tree.
        ud_misc (Dict[str, List[str]]): Miscellaneous features. Used in
        CoNLL-U Format. Refer https://universaldependencies.org/format.html
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.pos: str
        self.ud_xpos: str
        self.lemma: str
        self.chunk: str
        self.ner: str
        self.sense: str
        self.is_root: bool
        self.features: Dict[str, List[str]]
        self.ud_misc: Dict[str, List[str]]


class Sentence(Annotation):
    """
    A span based annotation :class:`Sentence`.

    Args:
        pack (DataPack): The data pack this token belongs to.
        begin (int): The offset of the first character in the sentence.
        end (int): The offset of the last character in the sentence + 1.
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class Document(Annotation):
    """
    A span based annotation :class:`Document`.

    Args:
        pack (DataPack): The data pack this token belongs to.
        begin (int): The offset of the first character in the document.
        end (int): The offset of the last character in the document + 1.
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class EntityMention(Annotation):
    """
    A span based annotation :class:`EntityMention`.

    Args:
        pack (DataPack): The data pack this token belongs to.
        begin (int): The offset of the first character in the entity mention.
        end (int): The offset of the last character in the entity mention + 1.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.ner_type: Optional[str] = None


class PredicateArgument(Annotation):
    """
    A span based annotation :class:`PredicateArgument`.

    Args:
        pack (DataPack): The data pack this token belongs to.
        begin (int): The offset of the first character in the predicate
            argument.
        end (int): The offset of the last character in the predicate argument
            + 1.
    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class PredicateMention(Annotation):
    """
    A span based annotation :class:`PredicateMention`.

    Args:
        begin (int): The offset of the first character in the predicate mention.
        end (int): The offset of the last character in the predicate mention
            + 1.
    """

    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class PredicateLink(Link):
    """
    A :class:`Link` type entry which take :class:`PredicateMention` as parent
    and :class:`PredicateArgument` as child.

    Args:
        parent (Entry, optional): the parent entry of the link.
        child (Entry, optional): the child entry of the link.
    """

    ParentType = PredicateMention
    """The entry type of the parent node of :class:`PredicateLink`."""

    ChildType = PredicateArgument
    """The entry type of the child node of :class:`PredicateLink`."""

    def __init__(self,
                 pack: DataPack,
                 parent: Optional[PredicateMention] = None,
                 child: Optional[PredicateArgument] = None):
        super().__init__(pack, parent, child)
        self.arg_type = None


# pylint: disable=duplicate-bases
class CoreferenceGroup(Group):
    """
    A :class:`Group` type entry which take :class:`EntityMention` as
    members.

    Args:
        members (Set[EntityMention], optional): a set of
            :class:`EntityMention` objects which are the members of the
            group.
    """
    MemberType = EntityMention
    """The entry type of group members of :class:`CoreferenceGroup`."""

    def __init__(
            self,
            pack: DataPack,
            members: Optional[Set[Entry]] = None
    ):
        super().__init__(pack, members)


class Dependency(Link):
    """
    A :class:`Link` type entry which represent a syntactic dependency.
    """
    ParentType = Token
    """The entry type of the parent node of :class:`Dependency`."""

    ChildType = Token
    """The entry type of the child node of :class:`Dependency`."""

    def __init__(self,
                 pack: DataPack,
                 parent: Optional[Token] = None,
                 child: Optional[Token] = None):
        super().__init__(pack, parent, child)
        self.dep_label = None
        self.rel_type: str
        self.dep_type: str


class RelationLink(Link):
    """
    A :class:`~ft.onto.base.Link` type entry which takes
    :class:`~ft.onto.base_ontology.EntityMention` objects as parent and child.

    Args:
        pack (DataPack): the containing pack of this link.
        parent (Entry, optional): the parent entry of the link.
        child (Entry, optional): the child entry of the link.
    """
    ParentType = EntityMention
    """The entry type of the parent node of :class:`RelationLink`."""
    ChildType = EntityMention
    """The entry type of the child node of :class:`RelationLink`."""

    def __init__(
            self,
            pack: DataPack,
            parent: Optional[EntityMention] = None,
            child: Optional[EntityMention] = None):
        super().__init__(pack, parent, child)
        self.rel_type = None


class Utterance(Annotation):
    r"""An annotation based entry useful for dialogue.

    Args:
        pack (DataPack): The data pack this token belongs to.
        begin (int): The offset of the first character in the entity mention.
        end (int): The offset of the last character in the entity mention + 1.

    """
    def __init__(self, pack: DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.seq_num: str
