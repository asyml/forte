"""
The reader that reads CoNLL ner_data into our internal json data format.
"""
import codecs
import os
from typing import Iterator, no_type_check

from nlp.pipeline.data.ontology import conll03_ontology
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.readers.file_reader import MonoFileReader

__all__ = [
    "CoNLL03Reader"
]


class CoNLL03Reader(MonoFileReader):
    """:class:`CoNLL03Reader` is designed to read in the CoNLL03-NER dataset.

    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``dataset_iterator()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``dataset_iterator()`` returns a list.
    """
    @no_type_check
    def __init__(self, lazy: bool = True):
        super().__init__(lazy)
        self.ner_ontology = conll03_ontology
        self.output_info = {
            self.ner_ontology.Sentence: ["span"],
            self.ner_ontology.Token: ["span", "chunk_tag", "pos_tag", "ner_tag"]
        }

    @staticmethod
    def dataset_path_iterator(dir_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory containing
        CONLL-formatted files.
        """
        for root, _, files in os.walk(dir_path):
            for data_file in files:
                if data_file.endswith("conll"):
                    yield os.path.join(root, data_file)

    def _read_document(self, file_path: str) -> DataPack:

        doc = codecs.open(file_path, "r", encoding="utf8")

        text = ""
        offset = 0
        has_rows = False

        sentence_begin = 0
        sentence_cnt = 0

        for line in doc:
            line = line.strip()

            if line != "" and not line.startswith("#"):
                conll_components = line.split()
                word = conll_components[1]
                pos_tag = conll_components[2]
                chunk_id = conll_components[3]
                ner_tag = conll_components[4]

                word_begin = offset
                word_end = offset + len(word)

                # add tokens
                kwargs_i = {"pos_tag": pos_tag, "chunk_tag": chunk_id,
                            "ner_tag": ner_tag}
                token = self.ner_ontology.Token(  # type: ignore
                    word_begin, word_end
                )

                token.set_fields(**kwargs_i)
                self.current_datapack.add_or_get_entry(token)

                text += word + " "
                offset = word_end + 1
                has_rows = True

            else:
                if not has_rows:
                    # skip consecutive empty lines
                    continue
                # add sentence
                sent = self.ner_ontology.Sentence(  # type: ignore
                    sentence_begin, offset - 1
                )
                self.current_datapack.add_or_get_entry(sent)

                sentence_begin = offset
                sentence_cnt += 1
                has_rows = False

        self.current_datapack.set_text(text)
        self.current_datapack.meta.doc_id = file_path
        doc.close()
        return self.current_datapack
