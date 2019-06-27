import collections
import numpy as np
from typing import Dict, List, Optional, Any
from nlp.pipeline.processors.base_processor import BaseProcessor
from nlp.pipeline.io.data_pack import DataPack


class VocabularyProcessor(BaseProcessor):
    """
    Build vocabulary from the input DataPack,
    Write the result to either:
        1. Another file
        2. write into the DataPack directly? Then if we want to build a joint
         vocabulary multiple Datapacks, we need to extract these vocabulary from
         datapacks, and then where to write this?
    Format: token, count
    """

    def __init__(self, min_frequency, max_vocab_size) -> None:
        super().__init__()
        self.min_frequency = min_frequency
        self.max_vocab_size = max_vocab_size


    def process(self, input_dict: Dict[str, Any], return_type) -> Dict[str,
                                                                       Any]:
        """
        Defines the process step of the processor.

        Args:
            input_dict

        Returns:
            output_dict
        """

        # Check the existence of required entries and fields
        words: List[str] = []
        for fn in input_dict:
            words += read_words(fn, newline_token=newline_token)

        # Note(haoransh): the following logic is copied from texar
        # https://github.com/asyml/texar-pytorch/blob/f7c79adfe64506995e4a1d40392dd06d84adf5ab/texar/data/data_utils.py

        counter = collections.Counter(words)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, counts = list(zip(*count_pairs))
        words: List[str]
        counts: List[int]
        if self.max_vocab_size >= 0:
            words = words[:self.max_vocab_size]
        counts = counts[:self.max_vocab_size]

        if return_type == "list":
            if not return_count:
                return words
            else:
                return words, counts
        elif return_type == "dict":
            word_to_id = dict(zip(words, range(len(words))))
            if not return_count:
                return word_to_id
            else:
                word_to_count = dict(zip(words, counts))

    return word_to_id, word_to_count

