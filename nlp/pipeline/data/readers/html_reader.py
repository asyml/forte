"""
The reader that reads plain text data into Datapacks.
"""
import os
from typing import Iterator, List, Union, Tuple
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
                       filepath: str,
                       replace_ops: List[Tuple[Union[Span, Pattern], str]] = None,
                       ) -> DataPack:
        pack = DataPack()
        with open(filepath, "r", encoding="utf8", errors='ignore') as file:
            text = file.read()

        text, _ = self.replace(text, replace_ops)

        document = self._ontology.Document(0, len(text))
        pack.add_or_get_entry(document)
        pack.set_text(text)
        pack.meta.doc_id = filepath
        return pack

    @staticmethod
    def replace(text: str, replace_ops: List[Tuple[Union[Span, Pattern], str]] = None):
        if replace_ops is None:
            return text, []

        # Converting regex in replace_ops to spans
        span_ops = []
        for op, replacement in replace_ops:
            spans = [Span(result.start(), result.end()) for result in op.finditer(text)] \
                if isinstance(op, Pattern) else [op]
            replacements = [replacement] * len(spans)
            span_ops.extend(list(zip(spans, replacements)))

        # Sorting the spans such that the order of replacement strings is maintained - utilizing the stable sort
        # property of python sort
        span_ops.sort(key=lambda item: item[0])

        if len(span_ops) == 0:
            return text, []

        # The spans should be mutually exclusive
        inverse_ops = []
        increment = 0
        prev_span_end = 0
        for span, replacement in span_ops:
            if span.begin < prev_span_end:
                raise ValueError("The replacement spans should be mutually exclusive")
            span_begin = span.begin + increment
            span_end = span.end + increment
            original_span_text = text[span_begin: span_end]
            text = text[:span_begin] + replacement + text[span_end:]
            increment += len(replacement) - (span.end - span.begin)
            replacement_span = Span(span_begin, span_begin + len(replacement))
            inverse_ops.append((replacement_span, original_span_text))
            prev_span_end = span.end

        return text, inverse_ops

