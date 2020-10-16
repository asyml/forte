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
Class for back translation op. This file also wraps a machine translation
model for the back translation. For simplicity, the model is not wrapped in
a processor.
"""
from typing import List
from transformers import MarianMTModel, MarianTokenizer
from forte.processors.data_augment.algorithms.text_replacement_op \
    import TextReplacementOp
from forte.data.ontology.core import Entry
from forte.common.configuration import Config


__all__ = [
    "BackTranslationOp",
]


class MarianMachineTranslator:
    r"""
    This class is a wrapper for the Marian Machine Translator
    (https://huggingface.co/transformers/model_doc/marian.html).
    Please refer to their doc for supported languages.
    """
    def __init__(self, src_lang: str = 'en', tgt_lang: str = 'fr'):
        self.src_lang: str = src_lang
        self.tgt_lang: str = tgt_lang
        self.model_name: str = 'Helsinki-NLP/opus-mt-{src}-{tgt}'.format(
            src=src_lang, tgt=tgt_lang
        )
        self.tokenizer = None
        self.model = None

    def initialize(self, src_lang: str = 'en', tgt_lang: str = 'fr'):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model_name = 'Helsinki-NLP/opus-mt-{src}-{tgt}'.format(
            src=src_lang, tgt=tgt_lang
        )
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)

    def translate(self, src_texts: List[str]) -> List[str]:
        translated: List[str] = self.model.generate(
            **self.tokenizer.prepare_seq2seq_batch(src_texts)
        )
        tgt_texts: List[str] = [
            self.tokenizer.decode(t, skip_special_tokens=True)
            for t in translated
        ]
        return tgt_texts


class BackTranslationOp(TextReplacementOp):
    r"""
    This class is a replacement op using back translation
    to generate data with the same semantic meanings. The
    input is translated to another language, then translated
    back to the original language, with pretrained
    machine-translation models.
    """
    def __init__(self, model_to, model_back, configs: Config):
        super().__init__(configs)
        self.model_to = model_to
        self.model_back = model_back

        self.model_to.initialize(
            configs['src_language'],
            configs['tgt_language']
        )
        self.model_back.initialize(
            configs['tgt_language'],
            configs['src_language']
        )

    def replace(self, input: Entry) -> str:
        r"""
        This function replaces a piece of text with back translation.
        Args:
            input: a string, could be a word, sentence or document.
        Returns:
            a string with a similar semantic meaning.
        """
        intermediate_texts: List[str] = self.model_to.translate([input.text])
        return self.model_back.translate(intermediate_texts)[0]
