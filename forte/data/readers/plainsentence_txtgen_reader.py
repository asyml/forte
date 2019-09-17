"""
The reader that reads CoNLL ner_data into our internal json data format.
"""
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.readers.file_reader import MonoFileReader

__all__ = [
    "PlainSentenceTxtgenReader"
]


class PlainSentenceTxtgenReader(MonoFileReader):
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
        self.define_output_info()

    def define_output_info(self):
        self.output_info = {
            self._ontology.Sentence: [],
        }

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

        output_pack = DataPack(name=output_pack_name)

        m_pack.update_pack(
            {
                input_pack_name: input_pack,
                output_pack_name: output_pack
            }
        )

        return m_pack
