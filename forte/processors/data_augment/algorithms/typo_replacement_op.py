import random
from typing import Tuple, Union, Dict, Any

from forte.data.ontology import Annotation
from forte.processors.data_augment.algorithms.text_replacement_op import (
    TextReplacementOp,
)
from forte.common.configuration import Config
from forte.processors.data_augment.algorithms.typo_generator import TypoGenerator

__all__ = [
    "TypoReplacementOp",
]


class TypoReplacementOp(TextReplacementOp):
    r"""
    This class is a replacement op using a pre-defined
    spelling mistake dictionary to simulate spelling mistake.

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

    def __init__(
        self, typoGenerator: TypoGenerator, configs: Union[Config, Dict[str, Any]]
    ):
        super().__init__(configs)
        self.typoGenerator = typoGenerator
        self.dict_path = configs['dict_path']

    def replace(self, input_anno: Annotation) -> Tuple[bool, str]:
        r"""
        This function replaces a word from a typo dictionary.

        Args:
            input_anno (Annotation): The input annotation.
        Returns:
            A tuple, where the first element is a boolean value indicating
            whether the replacement happens, and the second element is the
            replaced string.
        """
        # If the replacement does not happen, return False.
        if random.random() > self.configs.prob:
            return False, input_anno.text
        word: str = self.typoGenerator.generate(input_anno.text, self.dict_path)
        return True, word
        