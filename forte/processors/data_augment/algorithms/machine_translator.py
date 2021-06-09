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
This file wraps a machine translation model.
It could be used for back translation.
For simplicity, the model is not wrapped as a processor.
"""
from typing import List
from abc import abstractmethod
from transformers import MarianMTModel, MarianTokenizer

__all__ = [
    "MachineTranslator",
    "MarianMachineTranslator",
]


class MachineTranslator:
    r"""
    This class is a wrapper for machine translation models.

    Args:
        src_lang: The source language.
        tgt_lang: The target language.
        device: "cuda" for gpu, "cpu" otherwise.
    """

    def __init__(self, src_lang: str, tgt_lang: str, device: str):
        self.src_lang: str = src_lang
        self.tgt_lang: str = tgt_lang
        self.device = device

    @abstractmethod
    def translate(self, src_text: str) -> str:
        r"""
        This function translates the input text into target language.

        Args:
            src_text (str): The input text in source language.
        Returns:
            The output text in target language.
        """
        raise NotImplementedError


class MarianMachineTranslator(MachineTranslator):
    r"""
    This class is a wrapper for the Marian Machine Translator
    (https://huggingface.co/transformers/model_doc/marian.html).
    Please refer to their doc for supported languages.
    """

    def __init__(
        self, src_lang: str = "en", tgt_lang: str = "fr", device: str = "cpu"
    ):
        super().__init__(src_lang, tgt_lang, device)
        self.model_name = "Helsinki-NLP/opus-mt-{src}-{tgt}".format(
            src=src_lang, tgt=tgt_lang
        )
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.model = self.model.to(self.device)

    def translate(self, src_text: str) -> str:
        translated: List[str] = self.model.generate(
            # TODO: Should not use prepare_seq2seq_batch for deprecation
            **self.tokenizer.prepare_seq2seq_batch(
                # Have to use explicitly call `convert_to_tensors` to make
                # this line work in both transformers 3 and 4, probably won't
                # work in 5.
                [src_text]
            )
            .convert_to_tensors("pt")
            .to(self.device)
        )

        tgt_texts: List[str] = [
            self.tokenizer.decode(t, skip_special_tokens=True)
            for t in translated
        ]
        return tgt_texts[0]
