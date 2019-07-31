import json
import os
from collections import Counter
from typing import Any, Counter as CounterType, Dict, Iterator, List, Optional

import texar.torch as tx

from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.ontology import base_ontology
from nlp.pipeline.processors.base_processor import BaseProcessor
from nlp.pipeline.models.NER.utils import normalize_digit_word

__all__ = [
    "Alphabet",
    "VocabularyProcessor",
    "CoNLL03VocabularyProcessor",
]


class Alphabet:
    def __init__(
            self,
            name,
            word_cnt: Optional[CounterType[str]] = None,
            keep_growing: bool = True,
            ignore_case_in_query: bool = True,
            other_embeddings: Optional[Dict] = None,
    ):
        """

        :param name:
        :param keep_growing:
        :param ignore_case_in_query:
            If it's True, Alphabet will try to query the lowercased input from
            it's vocabulary if it cannot find the input in its keys.
        """
        self.__name = name
        self.reserved_tokens = tx.data.SpecialTokens

        self.instance2index: Dict = {}
        self.instances: List = []

        for sp in [
            self.reserved_tokens.PAD,
            self.reserved_tokens.BOS,
            self.reserved_tokens.EOS,
            self.reserved_tokens.UNK,
        ]:
            self.instance2index[sp] = len(self.instance2index)
            self.instances.append(sp)

        self.pad_id = self.instance2index[self.reserved_tokens.PAD]  # 0
        self.bos_id = self.instance2index[self.reserved_tokens.BOS]  # 1
        self.eos_id = self.instance2index[self.reserved_tokens.EOS]  # 2
        self.unk_id = self.instance2index[self.reserved_tokens.UNK]  # 3

        self.keep_growing = keep_growing
        self.ignore_case_in_query = ignore_case_in_query

        self.other_embeddings = other_embeddings

        if word_cnt is not None:
            for word in word_cnt:
                self.add(word)
            self.close()

    def add(self, instance):
        if instance not in self.instance2index:
            self.instance2index[instance] = len(self.instance2index)
            self.instances.append(instance)

    def get_index(self, instance):
        """
        :param instance: the input token
        :return: the index of the queried token in the dictionary
        """
        if instance is None:
            return self.instance2index[self.reserved_tokens.PAD]

        try:
            return self.instance2index[instance]
        except KeyError:
            if self.keep_growing:
                self.add(instance)
                return self.instance2index[instance]
            else:
                if self.ignore_case_in_query:
                    try:
                        return self.instance2index[instance.lower()]
                    except KeyError:
                        return self.instance2index[self.reserved_tokens.UNK]
                else:
                    return self.instance2index[self.reserved_tokens.UNK]

    def get_instance(self, index):
        try:
            return self.instances[index]
        except IndexError:
            raise IndexError("unknown index: %d" % index)

    def size(self):
        return len(self.instances)

    def items(self):
        return self.instance2index.items()

    def close(self):
        self.keep_growing = False

    def open(self):
        self.keep_growing = True

    def get_content(self):
        return {
            "instance2index": self.instance2index,
            "instances": self.instances,
        }

    def __from_json(self, data):
        self.instances = data["instances"]
        self.instance2index = data["instance2index"]

    def save(self, output_directory, name=None):
        """
        Save both alhpabet records to the given directory.
        :param output_directory: Directory to save model and weights.
        :param name: The alphabet saving name, optional.
        :return:
        """
        saving_name = name if name else self.__name

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        json.dump(
            self.get_content(),
            open(os.path.join(output_directory, saving_name + ".json"), "w"),
            indent=4,
        )

    def load(self, input_directory, name=None):
        """
        :param input_directory:
        :param name:
        :return:
        """
        loading_name = name if name else self.__name
        self.__from_json(
            json.load(
                open(os.path.join(input_directory, loading_name + ".json"))
            )
        )
        self.keep_growing = False


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

    def __init__(self, min_frequency) -> None:
        super().__init__()
        self.min_frequency = min_frequency

    def process(self,  # type: ignore
                input_pack: Iterator[DataPack]) -> Dict[str, Any]:
        """

        :param input_pack:
        :return:
        """
        pass


class CoNLL03VocabularyProcessor(VocabularyProcessor):
    """
    Vocabulary Processor for the format of CoNLL data
    Create the vocabulary for the word, character, pos tag, chunk id and ner
    tag
    """

    def __init__(
            self,
            min_frequency: int = -1,
            normalize_digit: bool = True,
    ) -> None:
        super().__init__(min_frequency)
        self.normalize_digit = normalize_digit

    def normalize_func(self, x):
        if self.normalize_digit:
            return normalize_digit_word(x)
        else:
            return x

    def process(  # type: ignore
            self,
            input_pack: Iterator[DataPack]) -> List[CounterType[str]]:
        """
        :param input_pack: The ner_data packs to create vocabulary with
        :return:
            A list of five counters for different ner_data features, for words,
            characters, POS tags, chunk IDs and Name Entity Recognition
        """
        word_cnt: Counter = Counter()
        char_cnt: Counter = Counter()
        pos_cnt: Counter = Counter()
        chunk_cnt: Counter = Counter()
        ner_cnt: Counter = Counter()

        for data_pack in input_pack:
            for instance in data_pack.get_data(
                    context_type="sentence",
                    requests={
                        base_ontology.Token:
                            ["chunk_tag", "pos_tag", "ner_tag"],
                    },
            ):
                for token in instance["Token"]["text"]:
                    for char in token:
                        char_cnt[char] += 1
                    word = self.normalize_func(token)
                    word_cnt[word] += 1

                for pos in instance["Token"]["pos_tag"]:
                    pos_cnt[pos] += 1
                for chunk in instance["Token"]["chunk_tag"]:
                    chunk_cnt[chunk] += 1
                for ner in instance["Token"]["ner_tag"]:
                    ner_cnt[ner] += 1

                # if a singleton is in pre-trained embedding dict,
                # set the count to min_occur + c

        for word in word_cnt:
            if word_cnt[word] < self.min_frequency:
                del word_cnt[word]

        return [word_cnt, char_cnt, pos_cnt, chunk_cnt, ner_cnt]
