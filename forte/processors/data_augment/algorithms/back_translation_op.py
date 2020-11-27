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
Class for back translation op. The input is translated
to another language, then translated back to the original language.
"""
from typing import Tuple
from forte.processors.data_augment.algorithms.text_replacement_op \
    import TextReplacementOp
from forte.data.ontology.core import Entry
from forte.common.configuration import Config
from forte.utils.utils import create_class_with_kwargs


__all__ = [
    "BackTranslationOp",
]


class BackTranslationOp(TextReplacementOp):
    r"""
    This class is a replacement op using back translation
    to generate data with the same semantic meanings. The
    input is translated to another language, then translated
    back to the original language, with pretrained
    machine-translation models.

    Args:
        configs (Config): The configuration for back translation model.
            It should have the following fields:

            - ``'src_lang'``:
                The source language of back translation.
            - ``'tgt_lang'``:
                The target language of back translation.
            - ``'model_to'``:
                The full qualified name of the model from source language
                to target language.
            - ``'model_back'``:
                The full qualified name of the model from target language
                to source language.
            - ``'device'``:
                "cpu" for the CPU or "cuda" for GPU.
    """
    def __init__(self, configs: Config):
        super().__init__(configs)
        self.model_to = create_class_with_kwargs(
            configs['model_to'],
            class_args={
                "src_lang": configs['src_language'],
                "tgt_lang": configs['tgt_language'],
                "device": configs["device"]
            }
        )
        self.model_back = create_class_with_kwargs(
            configs['model_back'],
            class_args={
                "src_lang": configs['tgt_language'],
                "tgt_lang": configs['src_language'],
                "device": configs["device"]
            }
        )

    def replace(self, input: Entry) -> Tuple[bool, str]:
        r"""
        This function replaces a piece of text with back translation.

        Args:
            input (Entry): An Entry, could be a word, sentence or document.

        Returns:
            A tuple, where the first element is a boolean value indicating
            whether the replacement happens, and the second element is the
            replaced string.
        """
        intermediate_text: str = self.model_to.translate(input.text)
        return True, self.model_back.translate(intermediate_text)
