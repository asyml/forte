# Copyright 2020 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processors that augment the data.
"""
import random
import string
from typing import List
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.processors.base.data_augment_processor \
    import ReplacementDataAugmentProcessor
from ft.onto.base_ontology import (
    Token, Sentence, Document, MultiPackLink
)


__all__ = [
    "TextGenerationDataAugmentProcessor"
]

random.seed(0)


class TextGenerationDataAugmentProcessor(ReplacementDataAugmentProcessor):
    r"""
    The data augmentation processor for text generation data.
    The multipack contains two datapacks: source and target.
    This class augment both datapacks and add the new datapacks to
    the original multipack as augmented source and target.
    """
    def new_pack(self):
        return MultiPack()

    def _process(self, multipack: MultiPack):
        r"""
        This function processes the source and target datapack separately,
        then insert the new datapacks to the multipack.
        Each pair of source and target document is linked with a MultiPackLink.
        """
        if not self.augmenter:
            raise KeyError("The processor has not been assigned an augmenter!")

        input_pack: DataPack = multipack.get_pack(
            self.configs.input_pack_name
        )
        output_pack: DataPack = multipack.get_pack(
            self.configs.output_pack_name
        )
        self._create_multipack_link(
            multipack,
            input_pack,
            output_pack
        )

        # Create aug_num pairs of augmented source & target datapacks
        for i in range(self.configs.aug_num):
            aug_input_pack: DataPack = self._process_pack(
                multipack.get_pack(self.configs.input_pack_name)
            )
            aug_output_pack: DataPack = self._process_pack(
                multipack.get_pack(self.configs.output_pack_name)
            )
            multipack.update_pack({
                "{}_{}".format(
                    self.configs.aug_input_pack_name,
                    str(i)
                ): aug_input_pack,
                "{}_{}".format(
                    self.configs.aug_output_pack_name,
                    str(i)
                ): aug_output_pack,
            })
            # Create a link between the source and target.
            self._create_multipack_link(
                multipack,
                aug_input_pack,
                aug_output_pack
            )

    def _create_multipack_link(
            self,
            multipack: MultiPack,
            src: DataPack,
            tgt: DataPack
    ):
        r"""
        Create a multipack link between the source and target document
        within a multipack.
        """
        src_docs: List[Document] = list(src.get(Document))
        tgt_docs: List[Document] = list(tgt.get(Document))
        # If the document is not included in the datapack, create a new one.
        if len(src_docs) == 0:
            src_docs.append(Document(src, 0, len(src.text)))
        if len(tgt_docs) == 0:
            tgt_docs.append(Document(tgt, 0, len(tgt.text)))

        # The datpack should only contains one document.
        assert len(src_docs) == 1
        assert len(tgt_docs) == 1
        src_doc: Document = src_docs[0]
        tgt_doc: Document = tgt_docs[0]
        # Link the source and target document.
        cross_link = MultiPackLink(multipack, src_doc, tgt_doc)
        multipack.add_entry(cross_link)

    def _process_pack(self, pack: DataPack) -> DataPack:
        r"""
        This function process a single datapack with an augmenter.
        It processes one sentence at a time.
        Args:
            pack: a datapack with original texts and annotations
        Returns:
            a datapack with augmented texts and annotations
        """
        data_pack: DataPack = DataPack()
        if len(pack.text) == 0:
            return data_pack
        pack_text: str = ""

        replacement_level: str = self.configs.replacement_level
        replacement_prob: float = self.configs.replacement_prob

        for sent in pack.get(Sentence):
            sent_text: str = sent.text
            sent_text_: str = "" # new sentence text
            # Replace the whole sentence.
            if replacement_level == 'sentence':
                if random.random() < replacement_prob:
                    sent_text = self.augmenter.augment(sent_text)
            # Replace each words.
            elif replacement_level == 'word':
                # The Token must be built in the preceeding pipeline.
                if len(list(pack.get(Token))) == 0 and len(pack.text) > 0:
                    raise KeyError("Tokens not found in the pack!")

                # Get the Tokens.
                tokens: List[Token] = list(pack.get(Token, sent))
                # Replace the words with the augmenter.
                for i, token in enumerate(tokens):
                    new_token_text: str = token.text
                    if token.text not in string.punctuation \
                            and random.random() < replacement_prob:
                        pos_tag: str = token.pos if token.pos else ''
                        new_token_text = self.augmenter.augment(
                            token.text,
                            pos_tag
                        )

                    # Get the gap span(might be spaces between tokens)
                    # between two tokens.
                    gap_begin: int = token.end
                    gap_end: int = tokens[i + 1].begin if i < len(tokens) - 1 \
                        else len(sent_text)
                    gap_text: str = pack.text[gap_begin:gap_end]

                    # Get the span of the new token.
                    new_token_begin: int = len(pack_text) + len(sent_text_)
                    sent_text_ += new_token_text
                    new_token_end: int = len(pack_text) + len(sent_text_)
                    Token(data_pack, new_token_begin, new_token_end)
                    # Append the gap text.
                    sent_text_ += gap_text
                sent_text = sent_text_

            elif replacement_level == 'character':
                for char in sent_text:
                    if random.random() < replacement_prob \
                        and char not in string.punctuation and char != ' ':
                        char = self.augmenter.augment(char)
                    sent_text_ += char
                sent_text = sent_text_

            # Build the pack text and sentence annotation.
            start_index: int = len(pack_text)
            pack_text += sent_text
            if "Sentence" in self.configs.augment_entries:
                Sentence(data_pack, start_index, len(pack_text))
            pack_text += " "

        pack_text = pack_text[:-1]
        # Build the Document annotation
        if "Document" in self.configs.augment_entries:
            Document(data_pack, 0, len(pack_text))
        data_pack.set_text(pack_text)
        return data_pack

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config.update({
            'input_pack_name': 'input_src',
            'output_pack_name': 'output_tgt',
            'aug_input_pack_name': 'aug_input_src',
            'aug_output_pack_name': 'aug_output_tgt',
        })
        return config
