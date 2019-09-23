"""
The reader that reads CoNLL ner_data into our internal json data format.
"""
import codecs
import os
from typing import Iterator, Any
from forte.data.io_utils import dataset_path_iterator
from forte.data.ontology import conll03_ontology
from forte.data.data_pack import DataPack
from forte.data.readers import PackReader

__all__ = [
    "CoNLL03Reader"
]


class CoNLL03Reader(PackReader):
    """:class:`CoNLL03Reader` is designed to read in the CoNLL03-NER dataset.

    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``iter()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``iter()`` returns a list.
    """

    def __init__(self, lazy: bool = True):
        super().__init__(lazy)
        self._ontology = conll03_ontology
        self.define_output_info()

    def define_output_info(self):
        self.output_info = {
            self._ontology.Document: [],
            self._ontology.Sentence: [],
            self._ontology.Token: ["chunk_tag", "pos_tag", "ner_tag"]
        }

    # pylint: disable=no-self-use
    def _collect(self, conll_directory) -> Iterator[Any]:  # type: ignore
        """
        Iterator over conll files in the data_source
        :param conll_directory: directory to the conll files.
        :return: Iterator over files with conll path
        """
        return dataset_path_iterator(conll_directory, "conll")

    def _cache_key_function(self, conll_file: str) -> str:
        return os.path.basename(conll_file)

    def parse_pack(self, file_path: str) -> DataPack:
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

                # add tokens
                kwargs_i = {"pos_tag": pos_tag, "chunk_tag": chunk_id,
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
                    # skip consecutive empty lines
                    continue
                # add sentence
                sent = self._ontology.Sentence(  # type: ignore
                    pack, sentence_begin, offset - 1
                )
                pack.add_or_get_entry(sent)

                sentence_begin = offset
                sentence_cnt += 1
                has_rows = False

        document = self._ontology.Document(pack, 0, len(text))  # type: ignore
        pack.add_or_get_entry(document)

        pack.set_text(text, replace_func=self.text_replace_operation)
        pack.meta.doc_id = file_path
        doc.close()
        return pack
