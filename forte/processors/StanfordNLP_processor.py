from typing import List

from texar.torch import HParams
import stanfordnlp

import forte.data.ontology.stanfordnlp_ontology as ontology
from forte.processors.base import PackProcessor, ProcessInfo
from forte.data import DataPack
from forte.common.resources import Resources

__all__ = [
    "StandfordNLPProcessor",
]


class StandfordNLPProcessor(PackProcessor):
    def __init__(self, models_path: str):
        super().__init__()
        self._ontology = ontology
        self.processors = ""
        self.nlp = None
        self.MODELS_DIR = models_path
        self.lang = 'en'  # English is default

    def set_up(self):
        stanfordnlp.download(self.lang, self.MODELS_DIR)

    # pylint: disable=unused-argument
    def initialize(self, configs: HParams, resource: Resources):
        self.processors = configs.processors
        self.lang = configs.lang
        self.set_up()
        self.nlp = stanfordnlp.Pipeline(**configs.todict(),
                                        models_dir=self.MODELS_DIR)

    def _define_input_info(self) -> ProcessInfo:
        input_info: ProcessInfo = {
            self._ontology.Document: ["span"]
        }
        return input_info

    def _define_output_info(self) -> ProcessInfo:
        """
        Define the output_info of the processor. This depends on the user's
        choice of processors for the StanfordNLP toolkit.
        Returns:

        """
        output_info: ProcessInfo = {self._ontology.Sentence: ["span"]}

        if "tokenize" in self.processors:
            token_outputs = ['span']
            if 'pos' in self.processors:
                token_outputs.append('pos_tag')
                token_outputs.append('upos')
                token_outputs.append('xpos')
            if 'lemma' in self.processors:
                token_outputs.append('lemma')
            output_info[self._ontology.Token] = token_outputs

        if 'depparse' in self.processors:
            output_info[self._ontology.Dependency] = \
                ["parent", "child", "rel_type"]

        return output_info

    @staticmethod
    def default_hparams():
        """
        This defines a basic Hparams structure for StanfordNLP.
        :return:
        """
        return {
            'processors': 'tokenize,pos,lemma,depparse',
            'lang': 'en',
            # Language code for the language to build the Pipeline
            'use_gpu': False,
        }

    def _process(self, input_pack: DataPack):
        doc = input_pack.text
        end_pos = 0

        # sentence parsing
        sentences = self.nlp(doc).sentences  # type: ignore

        # Iterating through stanfordnlp sentence objects
        for sentence in sentences:
            begin_pos = doc.find(sentence.words[0].text, end_pos)
            end_pos = doc.find(sentence.words[-1].text, begin_pos) + len(
                sentence.words[-1].text)
            sentence_entry = self._ontology.Sentence(
                input_pack, begin_pos, end_pos
            )
            input_pack.add_or_get_entry(sentence_entry)

            tokens: List[ontology.Token] = []
            if "tokenize" in self.processors:
                offset = sentence_entry.span.begin
                end_pos_word = 0

                # Iterating through stanfordnlp word objects
                for word in sentence.words:
                    begin_pos_word = sentence_entry.text. \
                        find(word.text, end_pos_word)
                    end_pos_word = begin_pos_word + len(word.text)
                    token = self._ontology.Token(
                        input_pack,
                        begin_pos_word + offset, end_pos_word + offset
                    )

                    if "pos" in self.processors:
                        token.set_fields(pos_tag=word.pos)
                        token.set_fields(upos=word.upos)
                        token.set_fields(xpos=word.xpos)

                    if "lemma" in self.processors:
                        token.set_fields(lemma=word.lemma)

                    tokens.append(token)
                    input_pack.add_or_get_entry(token)

            # For each sentence, get the dependency relations among tokens
            if "depparse" in self.processors:
                # Iterating through token entries in current sentence
                for token, word in zip(tokens, sentence.words):
                    child = token  # current token
                    parent = tokens[word.governor - 1]  # Root token
                    relation_entry = self._ontology.Dependency(
                        input_pack, parent, child)
                    relation_entry.set_fields(
                        rel_type=word.dependency_relation)

                    input_pack.add_or_get_entry(relation_entry)
