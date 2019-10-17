from functools import total_ordering


@total_ordering
class Span:
    """
    A class recording the span of annotations. :class:`Span` objects can
    be totally ordered according to their :attr:`begin` as the first sort key
    and :attr:`end` as the second sort key.

    Args:
        begin (int): The offset of the first character in the span.
        end (int): The offset of the last character in the span + 1. So the
            span is a left-closed and right-open interval ``[begin, end)``.
    """

    def __init__(self, begin: int, end: int):
        self.begin = begin
        self.end = end

    def __lt__(self, other):
        if self.begin == other.begin:
            return self.end < other.end
        return self.begin < other.begin

    def __eq__(self, other):
        return (self.begin, self.end) == (other.begin, other.end)
