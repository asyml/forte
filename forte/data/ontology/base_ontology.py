"""
This class defines the basic ontology supported by our system
"""
from typing import Optional, Set
from forte.data.ontology.top import Annotation, Link, Group

__all__ = [
    "Token",
    "Document",
    "EntityMention",
    "Sentence",
    "PredicateMention",
    "PredicateLink",
    "PredicateArgument",
    "CoreferenceGroup",
    "CoreferenceMention"
]


class Token(Annotation):
    """
    A span based annotation :class:`Token`.

    Args:
        begin (int): The offset of the first character in the token.
        end (int): The offset of the last character in the token + 1.
    """
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.pos_tag = None


class Sentence(Annotation):
    """
    A span based annotation :class:`Sentence`.

    Args:
        begin (int): The offset of the first character in the sentence.
        end (int): The offset of the last character in the sentence + 1.
    """
    pass


class Document(Annotation):
    """
    A span based annotation :class:`Document`.

    Args:
        begin (int): The offset of the first character in the document.
        end (int): The offset of the last character in the document + 1.
    """
    pass


class EntityMention(Annotation):
    """
    A span based annotation :class:`EntityMention`.

    Args:
        begin (int): The offset of the first character in the entity mention.
        end (int): The offset of the last character in the entity mention + 1.
    """
    def __init__(self, begin: int, end: int):
        super().__init__(begin, end)
        self.ner_type = None


class PredicateArgument(Annotation):
    """
    A span based annotation :class:`PredicateArgument`.

    Args:
        begin (int): The offset of the first character in the predicate
            argument.
        end (int): The offset of the last character in the predicate argument
            + 1.
    """
    pass


class PredicateMention(Annotation):
    """
    A span based annotation :class:`PredicateMention`.

    Args:
        begin (int): The offset of the first character in the predicate mention.
        end (int): The offset of the last character in the predicate mention
            + 1.
    """
    pass


class PredicateLink(Link):
    """
    A :class:`Link` type entry which take :class:`PredicateMention` as parent
    and :class:`PredicateArgument` as child.

    Args:
        parent (Entry, optional): the parent entry of the link.
        child (Entry, optional): the child entry of the link.
    """
    parent_type = PredicateMention
    """The entry type of the parent node of :class:`PredicateLink`."""
    child_type = PredicateArgument
    """The entry type of the child node of :class:`PredicateLink`."""

    def __init__(self,
                 parent: Optional[PredicateMention] = None,
                 child: Optional[PredicateArgument] = None):
        super().__init__(parent, child)
        self.arg_type = None


class CoreferenceMention(Annotation):
    """
    A span based annotation :class:`CoreferenceMention`.

    Args:
        begin (int): The offset of the first character in the coreference
            mention.
        end (int): The offset of the last character in the coreference mention
            + 1.
    """
    pass


class CoreferenceGroup(Group):
    """
    A :class:`Group` type entry which take :class:`CoreferenceMention` as
    members.

    Args:
        members (Set[CoreferenceMention], optional): a set of
            :class:`CoreferenceMention` objects which are the members of the
            group.
    """
    member_type = CoreferenceMention
    """The entry type of group members of :class:`CoreferenceGroup`."""

    def __init__(self, members: Optional[Set[CoreferenceMention]] = None):
        super().__init__(members)  # type: ignore
        self.coref_type = None
