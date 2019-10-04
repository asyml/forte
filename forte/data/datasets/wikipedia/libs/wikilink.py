"""Various extractors."""
import functools

import regex
from more_itertools import peekable
from typing import Callable, Iterable, Iterator, List, TypeVar, NamedTuple

from .common import CaptureResult, Span


class Section:
    """Section class."""
    def __init__(self, name: str, level: int, body: str):
        """Instantiate a section."""
        self.name = name
        self.level = level
        self.body = body
        self._full_body = None

    @property
    def is_preamble(self):
        """Return True when this section is the preamble of the page."""
        return self.level == 0

    @property
    def full_body(self) -> str:
        """Get the full body of the section."""
        if self._full_body is not None:
            return self._full_body

        if self.is_preamble:
            full_body = self.body
        else:
            equals = ''.join('=' for _ in range(self.level))
            full_body = '{equals}{name}{equals}\n{body}'.format(
                equals=equals,
                name=self.name,
                body=self.body,
            )
        self._full_body = full_body
        return full_body

    def __repr__(self):
        'Return a nicely formatted representation string'
        template = '{class_name}(name={name!r}, level={level!r}, '\
            'body={body!r})'
        return template.format(
            class_name=self.__class__.__name__,
            name=self.name,
            level=self.level,
            body=self.body[:20],
        )


section_header_re = regex.compile(
    r'''^
        (?P<equals>=+)              # Match the equals, greedy
        (?P<section_name>           # <section_name>:
            .+?                     # Text inside, non-greedy
        )
        (?P=equals)\s*              # Re-match the equals
        $
    ''', regex.VERBOSE | regex.MULTILINE)

templates_re = regex.compile(
    r'''
        \{\{
        (?P<content>(?s).*?)
        \}\}
    ''', regex.VERBOSE)


@functools.lru_cache(maxsize=1000)
def _pattern_or(words: List) -> str:
    words_joined = '|'.join(words)

    return r'(?:{})'.format(words_joined)


def references(source: str) -> Iterator[CaptureResult[str]]:
    """Return all the references found in the document."""
    pattern = regex.compile(
        r'''
            <ref
            .*?
            <\/ref>
        ''', regex.VERBOSE | regex.IGNORECASE | regex.DOTALL)

    for match in pattern.finditer(source):
        yield CaptureResult(match.group(0), Span(*match.span()))


def sections(source: str, include_preamble: bool=False) \
        -> Iterator[CaptureResult[Section]]:
    """Return the sections found in the document."""
    section_header_matches = peekable(section_header_re.finditer(source))
    if include_preamble:
        try:
            body_end = section_header_matches.peek().start()
            body_end -= 1  # Don't include the newline before the next section
        except StopIteration:
            body_end = len(source)
        preamble = Section(
            name='',
            level=0,
            body=source[:body_end],
        )
        yield CaptureResult(preamble, Span(0, body_end))

    for match in section_header_matches:
        name = match.group('section_name')
        level = len(match.group('equals'))

        body_begin = match.end() + 1  # Don't include the newline after
        try:
            body_end = section_header_matches.peek().start()
            body_end -= 1  # Don't include the newline before the next section
        except StopIteration:
            body_end = len(source)

        section = Section(
            name=name,
            level=level,
            body=source[body_begin:body_end],
        )

        yield CaptureResult(section, Span(match.start(), body_end))


# @functools.lru_cache(maxsize=10)
# @utils.listify
# def citations(source, language):
#     citation_synonyms = languages.citation[language]

#     citation_synonyms_pattern = _pattern_or(citation_synonyms)

#     pattern = regex.compile(
#         r'''
#             \{\{
#             \s*
#             %s
#             \s+
#             (?:(?s).*?)
#             \}\}
#         ''' % (citation_synonyms_pattern,), regex.VERBOSE
#     )

#     for match in pattern.finditer(source):
#         yield match.group(0)


def templates(source: str) -> Iterator[CaptureResult[str]]:
    """Return all the templates found in the document."""
    for match in templates_re.finditer(source):
        yield CaptureResult(match.group(0), Span(*match.span()))


class Wikilink:
    """Link class."""
    def __init__(self,
                 link: str,
                 anchor: str,
                 section_name: str,
                 section_level: int,
                 section_number: int):
        """Instantiate a link."""
        self.link = link
        self.anchor = anchor
        self.section_name = section_name
        self.section_level = section_level
        self.section_number = section_number

    def __repr__(self):
        """Return a nicely formatted representation string"""
        template = '{class_name}(link={link!r}, anchor={anchor!r})'
        return template.format(
            class_name=self.__class__.__name__,
            link=self.link,
            anchor=self.anchor,
        )

# See https://regex101.com/r/kF0yC9/12
# The text inside the 'link' group is title of the page, it is limited to 256
# chars since it is the max supported by MediaWiki for page titles [1].
# Furthermore pipes and brakets (|,[,]) are invalid characters for page
# titles [2]. Furthermore, newlines are not allowed [3].
# The anchor text allows pipes and closed brakets, but not open ones [3],
# newlines are allowed [3].
# See:
# [1] https://en.wikipedia.org/w/index.php?title=Wikipedia:Wikipedia_records\
#    &oldid=709472636#Article_with_longest_title
# [2] https://www.mediawiki.org/w/index.php?title=Manual:$wgLegalTitleChars\
#    &oldid=1274292
# [3] https://it.wikipedia.org/w/index.php?\
#   title=Utente:CristianCantoro/Sandbox&oldid=79784393#Test_regexp

wikilink_re = regex.compile(
    r'''\[\[                              # Match two opening brackets
       (?P<link>                          # <link>:
           [^\n\|\]\[\#\<\>\{\}]{0,256}   # Text inside link group
                                          # everything not illegal, non-greedy
                                          # can be empty or up to 256 chars
       )
       (?:                                # Non-capturing group
          \|                              # Match a pipe
          (?P<anchor>                     # <anchor>:
              [^\[]*?                     # Test inside anchor group:
                                          # match everything not an open braket
                                          # - non greedy
                                          # if empty the anchor text is link
          )
       )?                                 # anchor text is optional
       \]\]                               # Match two closing brackets
     ''', regex.VERBOSE | regex.MULTILINE)

SectionLimits = NamedTuple('SectionLimits', [
    ('name', str),
    ('level', int),
    ('number', int),
    ('begin', int),
    ('end', bool)
])


def wikilinks(source: str, sections: Iterator[CaptureResult[Section]]) \
        -> Iterator[CaptureResult[Wikilink]]:
    """Return the wikilinks found in the document."""
    wikilink_matches = peekable(wikilink_re.finditer(source, concurrent=True))

    sections_limits = [SectionLimits(name=section.name,
                                     level=section.level,
                                     number=idx,
                                     begin=span.begin,
                                     end=span.end)
                       for idx, (section, span) in enumerate(sections, 1)]

    last_section_seen = 0
    for match in wikilink_matches:
        link = match.group('link') or ''
        link = link.strip()
        anchor = match.group('anchor') or link
        # newlines in anchor are visualized as spaces.
        anchor = anchor.replace('\n', ' ').strip()

        link_start = match.start()

        link_section_number = 0
        link_section_name = '---~--- incipit ---~---'
        link_section_level = 0

        for section in sections_limits[last_section_seen:]:
            if section.begin <= link_start <= section.end:
                link_section_number = section.number
                link_section_name = section.name
                link_section_level = section.level
                last_section_seen = (link_section_number - 1)\
                    if link_section_number > 0 else 0
                break

        # For some reason if wikilink has no pipe, e.g. [[apple]] the regex
        # above captures everything in the anchor group, so we need to set
        # the link to the same page.
        if (anchor and not link):
            link = anchor

        wikilink = Wikilink(
            link=link,
            anchor=anchor,
            section_name=link_section_name,
            section_level=link_section_level,
            section_number=link_section_number
        )

        yield CaptureResult(wikilink, Span(link_start, match.end()))
