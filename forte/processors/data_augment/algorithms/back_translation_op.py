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
import random
from typing import Tuple

from forte.data.ontology import Annotation
from forte.processors.data_augment.algorithms.text_replacement_op import (
    TextReplacementOp,
)
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

    It will sample from a Bernoulli distribution to decide
    whether to replace the input, with `prob` as the probability
    of replacement.

    The configuration should have the following fields:

    Args:
        prob (float): The probability of replacement, should fall in [0, 1].
        src_lang (str): The source language of back translation.
        tgt_lang (str): The target language of back translation.
        model_to (str): The full qualified name of the model from
            source language to target language.
        model_back (str): The full qualified name of the model from
            target language to source language.
        device (str): "cpu" for the CPU or "cuda" for GPU.
    """

    def __init__(self, configs: Config):
        super().__init__(configs)
        self._validate_configs(configs)
        self.model_to = create_class_with_kwargs(
            configs["model_to"],
            class_args={
                "src_lang": configs["src_language"],
                "tgt_lang": configs["tgt_language"],
                "device": configs["device"],
            },
        )
        self.model_back = create_class_with_kwargs(
            configs["model_back"],
            class_args={
                "src_lang": configs["tgt_language"],
                "tgt_lang": configs["src_language"],
                "device": configs["device"],
            },
        )

    def _validate_configs(self, configs):
        prob = configs["prob"]
        if not prob or prob < 0 or prob > 1:
            raise ValueError("The prob should be a float between 0 and 1!")

        src_lang = configs["src_language"]
        if not src_lang or len(src_lang) == 0:
            raise ValueError("Please provide a valid source language!")

        tgt_lang = configs["tgt_language"]
        if not tgt_lang or len(tgt_lang) == 0:
            raise ValueError("Please provide a valid target language!")

        model_to = configs["model_to"]
        if not model_to or len(model_to) == 0:
            raise ValueError("Please provide a valid to-model!")

        model_back = configs["model_back"]
        if not model_back or len(model_back) == 0:
            raise ValueError("Please provide a valid back-model!")

        device = configs["device"]
        if device not in ("cpu", "cuda"):
            raise ValueError("The device must be 'cpu' or 'cuda'!")

    def replace(self, input_anno: Annotation) -> Tuple[bool, str]:
        r"""
        This function replaces a piece of text with back translation.

        Args:
            input_anno (Entry): An annotation, could be a word, sentence
              or document.

        Returns:
            A tuple, where the first element is a boolean value indicating
            whether the replacement happens, and the second element is the
            replaced string.
        """
        # If the replacement does not happen, return False.
        if random.random() > self.configs.prob:
            return False, input_anno.text
        intermediate_text: str = self.model_to.translate(input_anno.text)
        return True, self.model_back.translate(intermediate_text)
