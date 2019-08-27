"""The reader that reads prodigy text data with annotations into Datapacks."""

import json
from typing import Iterator, Any
from forte.data.ontology.base_ontology import Token, \
    Document, EntityMention
from forte.data.data_pack import DataPack
from forte.data.readers.file_reader import MonoFileReader

__all__ = [
    "ProdigyReader"
]


class ProdigyReader(MonoFileReader):
    """:class:`ProdigyTextReader` is designed to read in Prodigy output text
    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``iter()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``iter()`` returns a list.
    """

    def __init__(self, lazy: bool = False):
        super().__init__(lazy)
        self.define_output_info()

    def define_output_info(self):
        self.output_info = {
            Document: [],
            Token: [],
            EntityMention: ["ner_type"]
        }

    def _cache_key_function(self, collection):
        return str(collection)

    def _collect(self, **kwargs) -> Iterator[Any]:
        """
        Collects from Prodigy file path and returns an iterator
        of Prodigy annotation data. The elements in the iterator
        correspond to each line in the prodigy file.
        One element is expected to be parsed as one DataPack.
        :param data_source: str a Prodigy file path
        :yield: Iterator of each line in the prodigy file
        """
        with open(kwargs['data_source']) as f:
            for line in f:
                yield json.loads(line)

    def parse_pack(self, data: dict) -> DataPack:
        """
        Extracts information from input `data` of one document
        output from Prodigy Annotator including the text,
        tokens and its annotations into a DataPack
        :param data: a dict that contains information for one document
        :return pack: DataPack containing information extracted from `data`
        """
        single_doc = data
        pack = DataPack()
        text = single_doc['text']
        tokens = single_doc['tokens']
        spans = single_doc['spans']

        document = Document(0, len(text))
        pack.set_text(text, replace_func=self.text_replace_operation)
        pack.add_or_get_entry(document)

        for token in tokens:
            begin = token['start']
            end = token['end']
            token_entry = Token(begin, end)
            pack.add_or_get_entry(token_entry)

        for span_items in spans:
            begin = span_items['start']
            end = span_items['end']
            annotation_entry = EntityMention(begin, end)
            annotation_entry.ner_type = span_items['label']
            pack.add_or_get_entry(annotation_entry)

        pack.meta.doc_id = single_doc['meta']['id']

        return pack
