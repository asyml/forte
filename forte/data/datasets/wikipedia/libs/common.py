"""Classes for the extractors."""
from typing import Generic, NamedTuple, T

Identifier = NamedTuple("Identifier", [
    ('type', str),
    ('id', str),
])


class CaptureResult(NamedTuple('CaptureResult', [
    ('data', T),
    ('span', 'Span'),
]), Generic[T]):
    pass


class Span(NamedTuple('Span', [('begin', int), ('end', int)])):
    """Represent the begin and the end of a capture."""

    def __le__(self, other: 'Span') -> bool:
        # return self.begin >= other.begin and self.end <= other.end
        # HACK: the following is more efficient. Sorry :(
        return self[0] >= other[0] and self[1] <= other[1]

    def __lt__(self, other: 'Span') -> bool:
        return self[0] > other[0] and self[1] < other[1]
