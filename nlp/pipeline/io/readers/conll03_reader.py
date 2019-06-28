"""
The reader that reads CoNLL data into our internal json data format.
"""
import os
import logging
import codecs
from typing import Iterator
from nlp.pipeline.io.readers.file_reader import MonoFileReader
from nlp.pipeline.io.data_pack import DataPack
from nlp.pipeline.io.conll03_ontology import CoNLL03Ontology

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CoNLL03Reader(MonoFileReader):
    def __init__(self, lazy: bool = True):
        super().__init__(lazy)
        self.ner_ontology = CoNLL03Ontology

    @staticmethod
    def dataset_path_iterator(dir_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory containing
        CONLL-formatted files.
        """
        for root, _, files in os.walk(dir_path):
            for data_file in files:
                if data_file.endswith("gold_conll"):
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
                word_index_in_sent = conll_components[0]
                word = conll_components[1]
                pos_tag = conll_components[2]
                chunk_id = conll_components[3]
                ner_tag = conll_components[4]

                word_begin = offset
                word_end = offset + len(word)

                # add tokens
                kwargs_i = {"pos_tag": pos_tag, "chunk_tag": chunk_id,
                            "ner_tag": ner_tag}
                token = self.ner_ontology.Token(
                    self.component_name, word_begin, word_end
                )

                token.set_fields(**kwargs_i)
                self.current_datapack.add_entry(token)

                text += word + " "
                offset = word_end + 1
                has_rows = True

            else:
                if not has_rows:
                    # skip consecutive empty lines
                    continue
                # add sentence
                sent = self.ner_ontology.Sentence(
                    self.component_name, sentence_begin, offset-1
                )
                self.current_datapack.add_entry(sent)

                sentence_begin = offset
                sentence_cnt += 1
                has_rows = False

        self.current_datapack.text = text
        doc.close()
        return self.current_datapack

    def _record_fields(self):
        self.current_datapack.record_fields(
            [],
            self.component_name,
            self.ner_ontology.Sentence.__name__,
        )
        self.current_datapack.record_fields(
            ["chunk_tag", "pos_tag", "ner_tag"],
            self.component_name,
            self.ner_ontology.Token.__name__,
        )