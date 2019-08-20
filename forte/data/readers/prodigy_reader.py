"""The reader that reads prodigy text data with annotations into Datapacks."""

import json
from typing import Optional

from forte.data.ontology.base_ontology import Token, \
    Document, EntityMention
from forte.data.data_pack import DataPack, ReplaceOperationsType
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

    def _read_document(self, file_path: str,
                       replace_operations: Optional[ReplaceOperationsType]
                       ) -> DataPack:
        """
        Extracts the contents of a Prodigy data output (in JSON format) into
        a Datapack.
        :param file_path: json dictionary
        :return: DataPack object
        """
        pack = DataPack()

        with open(file_path) as data_file:
            single_doc = json.load(data_file)
            text = single_doc['text']
            tokens = single_doc['tokens']
            spans = single_doc['spans']

            document = Document(0, len(text))
            pack.set_text(text)
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

            pack.meta.doc_id = ""

        return pack
