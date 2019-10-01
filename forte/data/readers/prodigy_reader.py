"""The reader that reads prodigy text data with annotations into Datapacks."""

import json
from typing import Iterator, Any

from forte.data.data_pack import DataPack
from forte.data.ontology.base_ontology import Token, Document, EntityMention
from forte.data.readers.base_reader import PackReader

__all__ = [
    "ProdigyReader"
]


class ProdigyReader(PackReader):
    """:class:`ProdigyTextReader` is designed to read in Prodigy output text
    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``iter()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``iter()`` returns a list.
    """

    def define_output_info(self):
        # pylint: disable=no-self-use
        return {
            Document: [],
            Token: [],
            EntityMention: ["ner_type"]
        }

    # pylint: disable=no-self-use
    def _cache_key_function(self, data: dict) -> str:
        return data['meta']['id']

    # pylint: disable=no-self-use
    def _collect(self,  # type: ignore
                 prodigy_annotation_file: str) -> Iterator[Any]:
        """
        Collects from Prodigy file path and returns an iterator
        of Prodigy annotation data. The elements in the iterator
        correspond to each line in the prodigy file.
        One element is expected to be parsed as one DataPack.
        :param prodigy_annotation_file: a Prodigy file path
        :yield: Iterator of each line in the prodigy file
        """
        with open(prodigy_annotation_file) as f:
            for line in f:
                yield json.loads(line)

    def parse_pack(self, data: dict) -> DataPack:
        """
        Extracts information from input `data` of one document
        output from Prodigy Annotator including the text,
        tokens and its annotations into a DataPack.
        :param data: a dict that contains information for one document.
        :return pack: DataPack containing information extracted from `data`.
        """
        pack = DataPack()
        text = data['text']
        tokens = data['tokens']
        spans = data['spans']

        document = Document(pack, 0, len(text))
        pack.set_text(text, replace_func=self.text_replace_operation)
        pack.add_or_get_entry(document)

        for token in tokens:
            begin = token['start']
            end = token['end']
            token_entry = Token(pack, begin, end)
            pack.add_or_get_entry(token_entry)

        for span_items in spans:
            begin = span_items['start']
            end = span_items['end']
            annotation_entry = EntityMention(pack, begin, end)
            annotation_entry.set_fields(ner_type=span_items['label'])
            pack.add_or_get_entry(annotation_entry)

        pack.meta.doc_id = data['meta']['id']

        return pack
