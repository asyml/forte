"""
The reader for reading sentences from text files into MultiPack
"""
import os
from typing import Any, Iterator

from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.io_utils import dataset_path_iterator
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.readers.base_reader import MultiPackReader

from ft.onto.base_ontology import Sentence

__all__ = [
    "MultiPackSentenceReader"
]


class MultiPackSentenceReader(MultiPackReader):
    r"""
    :class:`PlainSentenceTxtgenReader` is designed to read a file where
    each line is a sentence, and wrap it with MultiPack for the following
    text generation processors.
    """

    # pylint: disable=attribute-defined-outside-init
    def initialize(self, resource: Resources, configs: HParams):
        self.resource = resource
        self.config = configs

    # pylint: disable=no-self-use
    def _collect(self, text_directory: str) -> Iterator[Any]:  # type: ignore
        return dataset_path_iterator(text_directory, '')

    # pylint: disable=no-self-use
    def _cache_key_function(self, txt_path: str) -> str:
        return os.path.basename(txt_path)

    def _parse_pack(self, file_path: str) -> Iterator[MultiPack]:

        m_pack: MultiPack = MultiPack()

        input_pack_name = self.config.input_pack_name
        output_pack_name = self.config.output_pack_name

        text = ""
        offset = 0
        with open(file_path, "r", encoding="utf8") as doc:

            input_pack = DataPack(doc_id=file_path)

            for line in doc:
                line = line.strip()

                if len(line) == 0:
                    continue

                # add sentence
                sent = Sentence(input_pack, offset, offset + len(line))
                input_pack.add_entry(sent)
                text += line + '\n'
                offset = offset + len(line) + 1

            input_pack.set_text(
                text, replace_func=self.text_replace_operation)

            output_pack = DataPack()

            m_pack.update_pack({
                input_pack_name: input_pack,
                output_pack_name: output_pack
            })

            yield m_pack

    @staticmethod
    def default_hparams():
        return {
            "input_pack_name": "input_src",
            "output_pack_name": "output_tgt"
        }
