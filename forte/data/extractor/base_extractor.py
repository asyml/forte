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


from abc import ABC
import logging
from typing import Tuple, List, Dict, Any, Union, Type, Hashable, Iterable
from ft.onto.base_ontology import Annotation
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.vocabulary import Vocabulary
from forte.data.converter.feature import Feature

logger = logging.getLogger(__name__)

__all__ = [
    "BaseExtractor"
]

class BaseExtractor(ABC):
    """The functionalities of this class are
        1. Build up vocabulary.
        2. Extract feature from datapack.
        3. Add prediction back to datapack.
    Args:
        config: an instance of Dict or forte.common.configuration.Config
            Required keys:
            "entry_type": Type[Entry], The ontology type where the feature from.
            "need_pad": bool, pass-in argument for building the vocabulary, this
                argument will also affect the behavior of "Converter".

            Optional keys:
            "vocab_method": str, pass-in argument for building the vocabulary,
                "raw", "indexing", "one-hot" are supported,
                default is "indexing".
            "vocab_use_unk": bool, pass-in argument for building the vocabulary,
                default is True.
    """
    def __init__(self, config: Union[Dict, Config]):

        self.config = Config(config, self.default_configs(), allow_new_hparam=True)

        if not hasattr(self.config, "entry_type"):
            logger.error("entry_type is required.")
            raise AttributeError("entry_type is required.")

        if self.config.vocab_method != "raw":
            self.vocab = Vocabulary(method=self.config.vocab_method,
                                    need_pad=self.config.need_pad,
                                    use_unk=self.config.vocab_use_unk)
        else:
            self.vocab = None

    @staticmethod
    def default_configs():
        return {
            "vocab_method": "indexing",
            "vocab_use_unk": True,
            "need_pad": True,
        }

    @property
    def state(self) -> Dict:
        return {
            "vocab_method": self.config.vocab_method,
            "need_pad": self.config.need_pad,
            "vocab_use_unk": self.config.vocab_use_unk,
            "entry_type": self.config.entry_type,
            "vocab": self.vocab.state if self.vocab else None,
        }
    
    @classmethod
    def from_state(cls, state: Dict):
        config = {
            "vocab_method": state["vocab_method"],
            "need_pad": state["need_pad"],
            "vocab_use_unk": state["vocab_use_unk"],
            "entry_type": state["entry_type"]
        }
        obj = cls(config)
        if "vocab" in state and state["vocab"] is not None:
            obj.vocab = Vocabulary.from_state(state["vocab"])
        return obj

    @property
    def entry_type(self) -> Type[Annotation]:
        return self.config.entry_type

    @property
    def vocab_method(self) -> str:
        return self.config.vocab_method

    def get_pad_value(self) -> Union[None, int, List[int]]:
        if self.vocab_method == "raw":
            return None
        else:
            return self.vocab.get_pad_value()

    def check_vocab(self):
        if not self.vocab:
            logger.error("When vocab_mehtod is raw, " \
                        "vocabulary is not built and " \
                        "operation on vocabulary should not" \
                        "be called.")
            raise AttributeError("Vocabulary does not exist.")

    def items(self) -> Iterable[Tuple[Hashable, int]]:
        self.check_vocab()
        return self.vocab.items()

    def size(self) -> int:
        self.check_vocab()
        return len(self.vocab)

    def add(self, element: Hashable):
        self.check_vocab()
        return self.vocab.add_element(element)

    def has_element(self, element: Hashable) -> bool:
        self.check_vocab()
        return self.vocab.has_element(element)

    def element2repr(self, element: Hashable) -> Union[int, List[int]]:
        self.check_vocab()
        return self.vocab.element2repr(element)

    def id2element(self, idx: int) -> Any:
        self.check_vocab()
        return self.vocab.id2element(idx)

    def get_dict(self) -> Dict[Hashable, int]:
        self.check_vocab()
        return self.vocab.get_dict()

    def predefined_vocab(self, predefined: set):
        """This function will add elements from the
        passed-in predefined set to the vocab. Different
        extractors might have different strategies to add
        these elements. Override this function if necessary.
        """
        for element in predefined:
            self.add(element)

    def update_vocab(self, pack: DataPack,
                    instance: Annotation):
        """This function will extract the feature from
        instance and add element in the feature to vocabulary.
        """
        raise NotImplementedError()

    def extract(self, pack: DataPack,
                instance: Annotation) -> Feature:
        """This function will extract feature from
        one instance in the pack.
        """
        raise NotImplementedError()

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Any):
        """This function will remove the original entry and
        add prediction to the pack.
        """
        raise NotImplementedError()
