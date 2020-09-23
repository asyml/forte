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
import nltk
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from typing import Iterable, Tuple, Dict
from abc import abstractmethod, ABC
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.common.resources import Resources
from forte.common.configuration import Config
from forte.processors.base.data_augment_processor import BaseDataAugmentProcessor
from ft.onto.base_ontology import (
    Token, Sentence, Document
)


__all__ = [
    "TextGenerationDataAugmentProcessor"
]

random.seed(0)

class TextGenerationDataAugmentProcessor(BaseDataAugmentProcessor):
    r"""
    The data augmentation processor for text generation data.
    The multipack contains two datapacks: source and target.
    This class augment both datapacks and add the new datapacks to
    the original multipack as augmented source and target.
    """
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.tokenizer = TreebankWordTokenizer()
        self.detokenizer = TreebankWordDetokenizer()
        try:
            nltk.pos_tag("apple")
        except:
            nltk.download('averaged_perceptron_tagger')

    def _process(self, multipack: MultiPack):
        r"""
        This function processes the source and target datapack separately,
        then insert the new datapacks to the multipack.
        """
        aug_input_pack: DataPack = self._process_pack(multipack.get_pack(self.configs.input_pack_name))
        aug_output_pack: DataPack = self._process_pack(multipack.get_pack(self.configs.output_pack_name))
        multipack.update_pack({
            self.configs.aug_input_pack_name: aug_input_pack,
            self.configs.aug_output_pack_name: aug_output_pack
        })

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
            # Replace the whole sentence.
            if replacement_level == 'sentence':
                if random.random() < replacement_prob:
                    sent_text = self.augmenter.augment(sent_text)
            # Replace each words.
            elif replacement_level == 'word':
                # Tokenize the sentence at first.
                tokens: List[str] = self.tokenizer.tokenize(sent_text)
                # Get the POS tags for synonym retreival.
                pos_tags: List[Tuple[str, str]] = nltk.pos_tag(tokens)
                for i, token in enumerate(tokens):
                    if token not in string.punctuation and random.random() < replacement_prob:
                        tokens[i] = self.augmenter.augment(token, {"pos_tag": pos_tags[i][1]})
                sent_text = self.detokenizer.detokenize(tokens)
            elif replacement_level == 'character':
                sent_text_: str = ""
                for char in sent_text:
                    if random.rand() < replacement_prob \
                        and char not in string.punctuation and char != ' ':
                        char = self.augmenter.augment(char)
                    sent_text_.append(char)
                sent_text = sent_text_

            # Build the pack text and sentence annotation.
            start_index: int = len(pack_text)
            pack_text += " " + sent_text if len(pack_text) > 0 else sent_text
            if "Sentence" in self.configs.augment_ontologies:
                Sentence(data_pack, start_index, len(pack_text))

        # Build the Document annotation
        if "Document" in self.configs.augment_ontologies:
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
            'replacement_prob': 0.1,
        })
        return config