"""
The reader that reads CoNLL ner_data into our internal json data datasets.
"""
import codecs
import logging
import os
from typing import Iterator, Any

from forte.data.data_pack import DataPack
from forte.data.io_utils import dataset_path_iterator
from forte.data.readers.base_reader import PackReader
from forte.data.ontology import conll03_ontology

__all__ = [
    "CoNLL03Reader"
]


class CoNLL03Reader(PackReader):
    """
    :class:`CoNLL03Reader` is designed to read in the CoNLL03-ner dataset.
    """

    def __init__(self):
        super().__init__()
        self._ontology = conll03_ontology

    # pylint: disable=no-self-use
    def _collect(self, conll_directory) -> Iterator[Any]:  # type: ignore
        """
        Iterator over conll files in the data_source

        Args:
            conll_directory:  directory to the conll files.

        Returns: Iterator over files in the path with conll extensions.
        """
        logging.info("Reading .conll from %s", conll_directory)
        return dataset_path_iterator(conll_directory, "conll")

    def _cache_key_function(self, conll_file: str) -> str:
        return os.path.basename(conll_file)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        pack = DataPack()
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

                # Add tokens.
                kwargs_i = {"pos_tag": pos_tag,
                            "chunk_tag": chunk_id,
                            "ner_tag": ner_tag}
                token = self._ontology.Token(  # type: ignore
                    pack, word_begin, word_end
                )

                token.set_fields(**kwargs_i)
                pack.add_or_get_entry(token)

                text += word + " "
                offset = word_end + 1
                has_rows = True
            else:
                if not has_rows:
                    # Skip consecutive empty lines.
                    continue
                # add sentence
                sent = self._ontology.Sentence(  # type: ignore
                    pack, sentence_begin, offset - 1
                )
                pack.add_or_get_entry(sent)

                sentence_begin = offset
                sentence_cnt += 1
                has_rows = False

        if has_rows:
            # Add the last sentence if exists.
            sent = self._ontology.Sentence(  # type: ignore
                pack, sentence_begin, offset - 1
            )
            sentence_cnt += 1
            pack.add_or_get_entry(sent)

        document = self._ontology.Document(pack, 0, len(text))  # type: ignore
        pack.add_or_get_entry(document)

        pack.set_text(text, replace_func=self.text_replace_operation)
        pack.meta.doc_id = file_path
        doc.close()

        yield pack
