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
from typing import Tuple, List
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.common.resources import Resources
from forte.common.configuration import Config
from forte.processors.base.data_augment_processor import ReplacementDataAugmentProcessor
from ft.onto.base_ontology import (
    Token, Sentence, Document, MultiPackLink
)


__all__ = [
    "RankingDataAugmentProcessor"
]

random.seed(0)

class RankingDataAugmentProcessor(ReplacementDataAugmentProcessor):
    r"""
    The data augmentation processor for ranking data.
    The input multipack contains query pack (and document packs).
    This class augments query and documents,
    then adds the augmented datapacks to the original multipack.
    """
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.tokenizer = TreebankWordTokenizer()
        self.detokenizer = TreebankWordDetokenizer()
        try:
            nltk.pos_tag("apple")
        except:
            nltk.download('averaged_perceptron_tagger')

    def new_pack(self):
        return multipack()

    def _process(self, input_pack: MultiPack):
        r"""
        This function processes query and documents according to user preference,
        then insert the new datapacks to the original multipack.
        Each pair of source and target document is linked with a MultiPackLink.
        """
        if not self.augmenter:
            raise KeyError("The processor has not been assigned an augmenter!")

        query_pack_name = self.config.query_pack_name
        query_pack = input_pack.get_pack(self.config.query_pack_name)
        # augment query
        if self.config.aug_query.lower() == 'true':
            aug_query_pack: DataPack = self._process_query_pack(query_pack)
            input_pack.update_pack({query_pack_name: aug_query_pack})

        # augment selected document
        if self.config.aug_document.lower() == 'true':
            packs = {}
            for doc_id in input_pack.pack_names:
                if doc_id == query_pack_name:
                    continue

                aug_doc_pack: DataPack = self._process_doc_pack(input_pack.get_pack(doc_id))
                # Todo: Does doc_id have to be int type to get elastic search works?
                packs['aug_' + doc_id] = aug_doc_pack

            input_pack.update_pack(packs)

    def _process_query_pack(self, pack: DataPack) -> DataPack:
        r"""
        This function process a single query datapack with an augmenter.
        It processes the query text by appending with augmented text.
        :param pack: a query datapack with original texts.
        :return: a query datapack with augmented texts.
        """
        data_pack: DataPack = DataPack()
        if len(pack.text) == 0:
            return data_pack

        query_text = pack.text
        sent_texts = nltk.sent_tokenize(query_text) # query text is usually one sentence.

        pack_text: str = ""
        for sent in sent_texts:
            sent_text: str = self.generate_augmented_sentences(sent)
            pack_text += " " + sent_text if len(pack_text) > 0 else sent_text

        new_query_text = query_text + pack_text
        data_pack.set_text(new_query_text)  # new_query_text = ori_query_text + aug_texts
        return data_pack

    def _process_doc_pack(self, pack: DataPack) -> DataPack:
        r"""
        This function process a single document datapack with an augmenter.
        It processes the document text (and annotation, if any).
        Note that for MSMacro dataset, the document datapack only has Document annotation and pack text defined.
        :param pack: a document datapack with original texts.
        :return: a document datapack with augmented texts.
        """
        data_pack: DataPack = DataPack()
        if len(pack.text) == 0:
            return data_pack

        if len(pack.get(Sentence)):
            sent_texts = pack.get(Sentence)
        else:
            doc_texts = pack.text
            sent_texts = nltk.sent_tokenize(doc_texts)

        pack_text: str = ""
        for sentt in sent_texts:
            sent = sentt.text
            sent_text: str = self.generate_augmented_sentence(sent)

            # Build the pack text and sentence annotation.
            start_index: int = len(pack_text)
            pack_text += " " + sent_text if len(pack_text) > 0 else sent_text
            if "Sentence" in self.configs.augment_entries:
                Sentence(data_pack, start_index, len(pack_text))

        # Build the Document annotation
        if "Document" in self.configs.augment_entries:
            Document(data_pack, 0, len(pack_text))

        data_pack.set_text(pack_text)
        return data_pack

    def generate_augmented_sent(self, sent_text: str) -> str:
        replacement_level: str = self.configs.replacement_level
        replacement_prob: float = self.configs.replacement_prob

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
                    tokens[i] = self.augmenter.augment(token, pos_tag = pos_tags[i][1])
            sent_text = self.detokenizer.detokenize(tokens)
        elif replacement_level == 'character':
            sent_text_: str = ""
            for char in sent_text:
                if random.rand() < replacement_prob \
                    and char not in string.punctuation and char != ' ':
                    char = self.augmenter.augment(char)
                sent_text_.append(char)
            sent_text = sent_text_

        return sent_text

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        config.update({
            "query_pack_name": "query",
            'aug_query': "true",
            'aug_document': "false"
        })
        return config
