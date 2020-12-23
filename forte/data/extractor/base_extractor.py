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
from typing import Tuple, Set, List, Dict, Any
from typing import Union, Type, Hashable, Iterable, Optional
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
    """The functionality of this class is as followed,
        1. Build vocabulary.
        2. Extract feature from datapack.
        3. Add prediction back to datapack.

    Args:
        config: An instance of `Dict` or
            :class:`forte.common.configuration.Config` that provides all
            configurable options. See :meth:`default_configs` for available
            options and default values.
            Required keys:
                entry_type: Type[Entry]. The ontology type that the extractor
                    get feature from.
    """
    def __init__(self, config: Union[Dict, Config]):

        self.config = Config(config, self.default_configs(),
                                allow_new_hparam=True)

        if not hasattr(self.config, "entry_type"):
            logger.error("entry_type is required.")
            raise AttributeError("entry_type is required.")

        if self.config.vocab_method != "raw":
            self.vocab: Optional[Vocabulary] = Vocabulary(
                                    method=self.config.vocab_method,
                                    need_pad=self.config.need_pad,
                                    use_unk=self.config.vocab_use_unk)
        else:
            self.vocab = None

    @staticmethod
    def default_configs():
        """
        Returns a dictionary of default hyper-parameters.

        "vocab_method": str
            What type of vocabulary is used for this extractor.
            "raw", "indexing", "one-hot" are supported, default is "indexing".
            Check the behavior of vocabulary under different setting
            in :class`forte.data.vocabulary.Vocabulary`

        "need_pad": bool
            Wether the <PAD> element should be added to vocabulary. And
            wether the feature need to be batched and paded. Default is True.

        "vocab_use_unk": bool
            Whether the <UNK> element should be added to vocabulary.
            Default is true.
        """
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
        if "vocab" in state and state["vocab"] is not None and \
            state["vocab"] != "None":
            obj.vocab = Vocabulary.from_state(state["vocab"])
        else:
            obj.vocab = None
        return obj

    @property
    def entry_type(self) -> Type[Annotation]:
        return self.config.entry_type

    @property
    def vocab_method(self) -> str:
        return self.config.vocab_method

    def get_pad_value(self) -> Union[None, int, List[int]]:
        if self.vocab:
            return self.vocab.get_pad_value()
        else:
            return None

    def items(self) -> Iterable[Tuple[Hashable, int]]:
        assert self.vocab
        return self.vocab.items()

    def size(self) -> int:
        assert self.vocab
        return len(self.vocab)

    def add(self, element: Hashable):
        assert self.vocab
        return self.vocab.add_element(element)

    def has_element(self, element: Hashable) -> bool:
        assert self.vocab
        return self.vocab.has_element(element)

    def element2repr(self, element: Hashable) -> Union[int, List[int]]:
        assert self.vocab
        return self.vocab.element2repr(element)

    def id2element(self, idx: int) -> Any:
        assert self.vocab
        return self.vocab.id2element(idx)

    def get_dict(self) -> Dict[Hashable, int]:
        assert self.vocab
        return self.vocab.get_dict()

    def predefined_vocab(self, predefined: Union[Set, List]):
        """This function will add elements from the
        passed-in predefined set to the vocab. Different
        extractors might have different strategies to add
        these elements. Override this function if necessary.

        Args:
            predefined (Union[Set, List]): A set or list contain
                elements to be added into the vocabulary.
        """
        for element in predefined:
            self.add(element)

    def update_vocab(self, pack: DataPack,
                    instance: Annotation):
        """This function will extract the feature from
        instance and add element in the feature to vocabulary.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will extractor feature.
        """
        raise NotImplementedError()

    def extract(self, pack: DataPack,
                instance: Annotation) -> Feature:
        """This function will extract feature from
        one instance in the pack.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will extractor feature.

        Returns:
            Feature: a feature that contains the extracted data.
        """
        raise NotImplementedError()

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Any):
        """This function will remove the original entry and
        add prediction to the pack.
        """
        raise NotImplementedError()
