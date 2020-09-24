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
Processors that augment the data for ranking tasks.
The input to this processor is a Multipack that contains a query only,
or contains a query and selected/ranked documents.
Users can choose to augment query text, or documents, or all.
Used in the pipeline that first creates a Multipack of query or query&docs.
After augmentation, the augmented query can be used to search for
a new sets documents, and the augmented documents can be re-ranked
through elastic search or bert processor.
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
from forte.processors.base.data_augment_processor import BaseDataAugmentProcessor
from ft.onto.base_ontology import (
    Token, Sentence, Document
)


__all__ = [
    "RankingDataAugmentProcessor"
]

random.seed(0)

class RankingDataAugmentProcessor(BaseDataAugmentProcessor):
    r"""
    The data augmentation processor for ranking data.
    The multipack contains query pack (and document packs).
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

    def _process(self, input_pack: MultiPack):
        r"""
        This function processes query and documents according to user preference,
        then insert the new datapacks to the original multipack.
        """
        query_pack_name = self.config.query_pack_name
        query_pack = input_pack.get_pack(self.config.query_pack_name)
        # query_entry = list(query_pack.get(Query))[0]

        if self.config.aug_query.lower() == 'true':
            aug_query_pack: DataPack = self._process_pack(query_pack)
            input_pack.update_pack({'aug_' + query_pack_name: aug_query_pack})

        if self.config.aug_document.lower() == 'true':
            packs = {}
            for doc_id in input_pack.pack_names:
                if doc_id == query_pack_name:
                    continue

                aug_pack: DataPack = self._process_pack(input_pack.get_pack(doc_id))
                packs['aug_' + doc_id] = aug_pack

            input_pack.update_pack(packs)

    def _process_pack(self, pack: DataPack):
        r"""
        This function process a single datapack with an augmenter.
        It processes one sentence at a time.
        :param pack: a datapack with original texts and annotations
        :return: a datapack with augmented texts and annotations

        Query datapack only sets text, documents only set Document and text
        """
        data_pack: DataPack = DataPack()
        if len(pack.text) == 0:
            return data_pack
        pack_text: str = ""

        replacement_level: str = self.configs.replacement_level
        replacement_prob: float = self.configs.replacement_prob

        doc_text = pack.text
        sent_texts = nltk.sent_tokenize(doc_text)
        for sent_text in sent_texts:
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
            "query_pack_name": "query",
            'aug_query': "true",
            'aug_document': "true"
        })
        return config