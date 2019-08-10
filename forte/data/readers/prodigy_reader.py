"""The reader that reads prodigy text data with annotations into Datapacks."""

import json
from typing import List
from forte.data import Token, \
    Document, EntityMention
from forte.data import DataPack
from forte.data import MonoFileReader

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

    @staticmethod
    def _iterable_json(file_path):
        """
        Iterates over the contents of a json file and yields
        decoded lines.
        :str file_path: path to the jsonl file
        :yield: json dictionary
        """
        with open(file_path) as f:
            for line in f:
                yield json.loads(line)

    @staticmethod
    def dataset_path_iterator(file_path: str):
        """
        Yields a single document as json dictionary
        :param file_path: path to the JSONL file
        :yield: json dict
        """
        data_iterable = ProdigyReader._iterable_json(file_path)

        for single_doc in data_iterable:
            yield single_doc

    def iter(self, file_path: str) -> List[DataPack]:
        """
        Reads a jsonl file into a list of DataPacks
        TODO add another lazy function that gives an Iterator
        :param dir_path:
        :return: List[DataPack]
        """
        data_iterable = self._iterable_json(file_path)

        DataPacks = []

        for single_doc in data_iterable:
            DataPacks.append(self.read(single_doc))

        return DataPacks

    def read(self, single_doc: dict) -> DataPack:  # type: ignore
        """
        Extracts the contents of a json dictionary into a Datapack
        :param single_doc: json dictionary
        :return: DataPack object
        """
        pack = DataPack()
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

    def _read_document(self, file_path: str) -> DataPack:
        # For some reason the mypy test needs this function to exist
        raise NotImplementedError
