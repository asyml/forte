"""
The reader that reads CoNLL ner_data into our internal json data format.
"""
import os
from typing import Iterator

from forte.data import DataPack
from forte.data import MultiPack
from forte.data import MonoFileMultiPackReader

__all__ = [
    "PlainSentenceTxtgenReader"
]


class PlainSentenceTxtgenReader(MonoFileMultiPackReader):
    """:class:`PlainSentenceTxtgenReader` is designed to read a file where
    each line is a sentence, and wrap it with MultiPack for the following
    text generation processors.

    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``dataset_iterator()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``dataset_iterator()`` returns a list.
    """

    def __init__(self, lazy: bool = True):
        super().__init__(lazy)
        self.current_datapack: MultiPack = MultiPack()
        self.define_output_info()

    def define_output_info(self):
        self.output_info = {
            self._ontology.Sentence: [],
        }

    @staticmethod
    def dataset_path_iterator(dir_path: str) -> Iterator[str]:
        """
        An iterator returning all file_paths in a directory.
        """
        for root, _, files in os.walk(dir_path):
            files.sort()
            for data_file in files:
                yield os.path.join(root, data_file)

    def _read_document(self, file_path: str) -> MultiPack:

        input_pack_name = "input_src"
        output_pack_name = "output_tgt"

        with open(file_path, "r", encoding="utf8") as doc:

            text = ""
            offset = 0

            sentence_cnt = 0

            input_pack = DataPack(doc_id=file_path, name=input_pack_name)

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

            input_pack.set_text(text)
            input_pack.meta.doc_id = file_path

        output_pack = DataPack(name=output_pack_name)

        self.current_datapack.update_pack(
            **{
                input_pack_name: input_pack,
                output_pack_name: output_pack
            }
        )

        return self.current_datapack
