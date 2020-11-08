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
    """
    def __init__(self, configs: Config):
        super().__init__(configs)
        self.model_to = create_class_with_kwargs(
            configs['model_to'],
            class_args={
                "src_lang": configs['src_language'],
                "tgt_lang": configs['tgt_language']
            }
        )
        self.model_back = create_class_with_kwargs(
            configs['model_back'],
            class_args={
                "src_lang": configs['tgt_language'],
                "tgt_lang": configs['src_language']
            }
        )

    def replace(self, input: Entry) -> Tuple[bool, str]:
        r"""
        This function replaces a piece of text with back translation.
        Args:
            input: a string, could be a word, sentence or document.
        Returns:
            a bool value indicating that the replacement happened.
            a string with a similar semantic meaning.
        """
        intermediate_text: str = self.model_to.translate(input.text)
        return True, self.model_back.translate(intermediate_text)
