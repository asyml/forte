"""
The reader that reads CoNLL ner_data into our internal json data datasets.
"""
import os
from typing import Any, Iterator

from forte.data.io_utils import dataset_path_iterator
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.readers.base_reader import MultiPackReader

__all__ = [
    "PlainSentenceTxtgenReader"
]


class PlainSentenceTxtgenReader(MultiPackReader):
    """
    :class:`PlainSentenceTxtgenReader` is designed to read a file where
    each line is a sentence, and wrap it with MultiPack for the following
    text generation processors.
    """
    def define_output_info(self):
        return {
            self._ontology.Sentence: [],
        }

    # pylint: disable=no-self-use
    def _collect(self, text_directory: str) -> Iterator[Any]:  # type: ignore
        return dataset_path_iterator(text_directory, '')

    # pylint: disable=no-self-use
    def _cache_key_function(self, txt_path: str) -> str:
        return os.path.basename(txt_path)

    def parse_pack(self, file_path: str) -> MultiPack:
        m_pack: MultiPack = MultiPack()

        input_pack_name = "input_src"
        output_pack_name = "output_tgt"

        with open(file_path, "r", encoding="utf8") as doc:
            text = ""
            offset = 0

            sentence_cnt = 0

            input_pack = DataPack(doc_id=file_path)

            for line in doc:
                line = line.strip()
                if len(line) == 0:
                    # skip empty lines
                    continue
                # add sentence
                sent = self._ontology.Sentence(  # type: ignore
                    offset, offset + len(line)
                )
                input_pack.add_entry(sent)
                text += line + '\n'
                offset = offset + len(line) + 1

                sentence_cnt += 1

                if sentence_cnt >= 20:
                    break

            input_pack.set_text(text, replace_func=self.text_replace_operation)

        output_pack = DataPack()

        m_pack.update_pack(
            {
                input_pack_name: input_pack,
                output_pack_name: output_pack
            }
        )

        return m_pack
