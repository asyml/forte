"""
This class defines the basic ontology supported by our system
"""
import typing
from forte.data import data_pack
from forte.data.ontology.base import top

__all__ = [
    "Token",
    "Sentence",
    "Document",
    "EntityMention",
    "PredicateArgument",
    "PredicateMention",
    "PredicateLink",
    "CoreferenceGroup",
    "CoreferenceMention",
    "Dependency",
    "RelationLink"
]


class Token(top.Annotation):
    """
    A span based annotation :class:`Token`.

    Args:
        pack (DataPack): The data pack this token belongs to.
        begin (int): The offset of the first character in the token.
        end (int): The offset of the last character in the token + 1.

    Attributes:
        xpos (str): Language specific pos tag.
        lemma (str): Lemma or stem of word form.
        is_root (bool): If the token is a root of, say, dependency tree.
    """

    def __init__(self, pack: data_pack.DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.pos: str
        self.xpos: str
        self.lemma: str
        self.chunk: str
        self.ner: str
        self.sense: str
        self.is_root: bool
        self.features: typing.Dict[str, typing.List[str]]
        self.miscellaneous: typing.Dict[str, typing.List[str]]


class Sentence(top.Annotation):
    """
    A span based annotation :class:`Sentence`.

    Args:
        pack (DataPack): The data pack this token belongs to.
        begin (int): The offset of the first character in the sentence.
        end (int): The offset of the last character in the sentence + 1.
    """
    def __init__(self, pack: data_pack.DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class Document(top.Annotation):
    """
    A span based annotation :class:`Document`.

    Args:
        pack (DataPack): The data pack this token belongs to.
        begin (int): The offset of the first character in the document.
        end (int): The offset of the last character in the document + 1.
    """
    def __init__(self, pack: data_pack.DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class EntityMention(top.Annotation):
    """
    A span based annotation :class:`EntityMention`.

    Args:
        pack (DataPack): The data pack this token belongs to.
        begin (int): The offset of the first character in the entity mention.
        end (int): The offset of the last character in the entity mention + 1.
    """

    def __init__(self, pack: data_pack.DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)
        self.ner_type: str


class PredicateArgument(top.Annotation):
    """
    A span based annotation :class:`PredicateArgument`.

    Args:
        pack (DataPack): The data pack this token belongs to.
        begin (int): The offset of the first character in the predicate
            argument.
        end (int): The offset of the last character in the predicate argument
            + 1.
    """
    def __init__(self, pack: data_pack.DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class PredicateMention(top.Annotation):
    """
    A span based annotation :class:`PredicateMention`.

    Args:
        begin (int): The offset of the first character in the predicate mention.
        end (int): The offset of the last character in the predicate mention
            + 1.
    """

    def __init__(self, pack: data_pack.DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class PredicateLink(top.Link):
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
                 pack: data_pack.DataPack,
                 parent: typing.Optional[PredicateMention] = None,
                 child: typing.Optional[PredicateArgument] = None):
        super().__init__(pack, parent, child)
        self.arg_type = None


class CoreferenceMention(top.Annotation):
    """
    A span based annotation :class:`CoreferenceMention`.

    Args:
        pack (DataPack): The data pack this token belongs to.
        begin (int): The offset of the first character in the coreference
            mention.
        end (int): The offset of the last character in the coreference mention
            + 1.
    """
    def __init__(self, pack: data_pack.DataPack, begin: int, end: int):
        super().__init__(pack, begin, end)


class CoreferenceGroup(top.Group):
    """
    A :class:`Group` type entry which take :class:`CoreferenceMention` as
    members.

    Args:
        members (Set[CoreferenceMention], optional): a set of
            :class:`CoreferenceMention` objects which are the members of the
            group.
    """
    MemberType = CoreferenceMention
    """The entry type of group members of :class:`CoreferenceGroup`."""

    def __init__(
            self,
            pack: data_pack.DataPack,
            members: typing.Optional[typing.Set[top.Entry]] = None,
    ):
        super().__init__(pack, members)


class Dependency(top.Link):
    """
    A :class:`Link` type entry which represent a syntactic dependency.
    """
    ParentType = Token
    """The entry type of the parent node of :class:`Dependency`."""

    ChildType = Token
    """The entry type of the child node of :class:`Dependency`."""

    def __init__(self,
                 pack: data_pack.DataPack,
                 parent: typing.Optional[Token] = None,
                 child: typing.Optional[Token] = None):
        super().__init__(pack, parent, child)
        self.dep_label = None
        self.rel_type: str
        self.dep_type: str


class RelationLink(top.Link):
    """
    A :class:`~forte.data.ontology.forte.data.ontology.base.Link` type entry
    which takes :class:`~forte.data.ontology.base_ontology.EntityMention`
    objects as parent and child.

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
            pack: data_pack.DataPack,
            parent: typing.Optional[EntityMention] = None,
            child: typing.Optional[EntityMention] = None):
        super().__init__(pack, parent, child)
        self.rel_type = None
