"""
The reader that reads plain text data into Datapacks.
"""
import codecs
import os
from typing import Iterator, Optional, List, Union
import re
from re import Pattern

from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.ontology import base_ontology
from nlp.pipeline.data.readers.file_reader import MonoFileReader
from nlp.pipeline.data.ontology.top import Span

__all__ = [
    "HTMLReader"
]


class HTMLReader(MonoFileReader):
    """:class:`HTMLReader` is designed to read in html file dataset.

    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``iter()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``iter()`` returns a list.
    """

    def __init__(self, lazy: bool = True):
        super().__init__(lazy)
        self._ontology = base_ontology
        self.define_output_info()

    def define_output_info(self):
        self.output_info = {
            self._ontology.Document: [],
        }

    @staticmethod
    def dataset_path_iterator(dir_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory containing
        .html or .htm files.
        """
        html_file_extensions = ['.html', '.htm']
        for root, _, files in os.walk(dir_path):
            files.sort()
            for data_file in files:
                for extension in html_file_extensions:
                    if data_file.endswith(extension):
                        yield os.path.join(root, data_file)
                        break

    def _read_document(self,
                       file_path: str,
                       replace_ops=[],
                       ) -> DataPack:
        pack = DataPack()
        doc = codecs.open(file_path, "rb", encoding="utf8", errors='ignore')
        text = doc.read()

        span_ops = []
        for op, replacement in replace_ops:
            spans = [Span(result.start(), result.end()) for result in op.finditer(text)] \
                if isinstance(op, Pattern) else [op]
            replacements = ['' if not replacement else replacement] * len(spans)
            span_ops.extend(list(zip(spans, replacements)))

        text, inverse_ops = self.replace(text, span_ops)

        document = self._ontology.Document(0, len(text))
        pack.add_or_get_entry(document)

        pack.set_text(text)
        pack.meta.doc_id = file_path
        doc.close()
        return text, inverse_ops, pack

    @staticmethod
    def replace(text, span_ops):
        # Assuming that the spans are mutually exclusive
        # TODO: check for the above assumption
        # Stable sort is required for it to work
        span_ops.sort(key=lambda item: item[0])
        inverse_ops = []
        increment = 0
        for span, replacement in span_ops:
            span_begin = span.begin + increment
            span_end = span.end + increment
            original_span_text = text[span_begin: span_end]
            text = text[:span_begin] + replacement + text[span_end:]
            increment += len(replacement) - (span.end - span.begin)
            replacement_span = Span(span_begin, span_begin + len(replacement))
            inverse_ops.append((replacement_span, original_span_text))

        return text, inverse_ops


if __name__ == "__main__":
    file_path = 'test.html'
    file_text = '<html>' \
                '<head><title>The Original Title </title></head>' \
                '<body>HTML web page contents </body></html>'
    with open(file_path, 'w') as f:
        f.write(file_text)

    reader = HTMLReader()
    span_ops = [(re.compile("</?[a-z]+>"), ''), (Span(19, 31), 'The Replaced')]
    text, inverse_ops, pack = reader._read_document(file_path, span_ops)
    assert text == 'The Replaced Title ' \
                   'HTML web page contents '

    orig_text, _ = HTMLReader.replace(text, inverse_ops)
    assert orig_text == file_text
