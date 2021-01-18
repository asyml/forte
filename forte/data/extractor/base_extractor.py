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
This file implements BaseExtractor, which is the abstract class other
extractors will inherit from.
"""
from abc import ABC
import logging
from typing import Tuple, Set, List, Dict, Any
from typing import Union, Type, Hashable, Iterable, Optional
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.data.vocabulary import Vocabulary
from forte.data.converter.feature import Feature

logger = logging.getLogger(__name__)

__all__ = [
    "BaseExtractor"
]


class BaseExtractor(ABC):
    r"""The functionality of Extractor is as followed. Most of
    the time, a user will not need to call this class explicitly,
    they will be called by the framework.

        1. Build vocabulary.
        2. Extract feature from datapack.
        3. Perform pre-evaluation action on datapack.
        4. Add prediction to datapack.

    Explanation:
        Vocabulary: Vocabulary is maintained as an inner class
            in extractor. It will store the mapping from element
            to index, which is an integer, and representation,
            which could be an index integer or one-hot vector
            depending on the configuration of the vocabulary.
            Check :class:`forte.data.vocabulary.Vocabulary` for
            more details.
        Feature: A feature basically wraps the data we want from
            one instance in a datapack. For example, the instance
            can be one sentence in a datapack. Then the data wrapped
            by the feature could be the token text of this sentence.
            The data is already converted as list of indexes using
            vocabulary. Besides the data, other information like the
            raw data before indexing and some meta_data will also be
            stored in the feature. Check
            :class:`forte.data.converter.feature.Feature` for more
            details.
        Remove feature / Add prediction: Removing feature means remove
            the existing data in the datapack. If we remove the feature
            in the pack, then extracting feature will return empty list.
            Adding prediction means we add the prediction from model
            back to the datapack. If a datapack has some old data (for
            example, the golden data in the test set), we can first
            remove those data and then add our model prediction to
            the pack.

    Args:
        config: An instance of `Dict` or
            :class:`forte.common.configuration.Config` that provides all
            configurable options. See :meth:`default_configs` for available
            options and default values. Entry_type is the key that need to
            be passed in and there will not be default value for this key.

            entry_type: Type[Entry]. Required. The ontology type that the
                extractor will get feature from.
    """
    _VOCAB_ERROR_MSG = "When vocab_method is raw, vocabulary " \
                       "will not be built. Functions operating " \
                       "on vocabulary should not be called."

    def __init__(self, config: Union[Dict, Config]):
        self.config = Config(config, self.default_configs())

        if self.config.entry_type is None:
            raise AttributeError("entry_type needs to be specified in "
                                 "the configuration of an extractor.")

        if self.config.vocab_method != "raw":
            self.vocab: Optional[Vocabulary] = \
                Vocabulary(method=self.config.vocab_method,
                           need_pad=self.config.need_pad,
                           use_unk=self.config.vocab_use_unk)
        else:
            self.vocab = None

    @classmethod
    def default_configs(cls):
        r"""Returns a dictionary of default hyper-parameters.

        "vocab_method": str
            What type of vocabulary is used for this extractor.
            "raw", "indexing", "one-hot" are supported, default is "indexing".
            Check the behavior of vocabulary under different setting
            in :class`forte.data.vocabulary.Vocabulary`

        "need_pad": bool
            Whether the <PAD> element should be added to vocabulary. And
            whether the feature need to be batched and padded. Default is True.

        "vocab_use_unk": bool
            Whether the <UNK> element should be added to vocabulary.
            Default is true.
        """
        return {
            "entry_type": None,
            "vocab_method": "indexing",
            "vocab_use_unk": True,
            "need_pad": True,
        }

    @property
    def entry_type(self) -> Type[Annotation]:
        return self.config.entry_type

    @property
    def vocab_method(self) -> str:
        return self.config.vocab_method

    def get_pad_value(self) -> Union[None, int, List[int]]:
        if self.vocab is not None:
            return self.vocab.get_pad_value()
        else:
            return None

    def items(self) -> Iterable[Tuple[Hashable, int]]:
        if self.vocab is None:
            raise AttributeError(self._VOCAB_ERROR_MSG)
        return self.vocab.items()

    def size(self) -> int:
        if self.vocab is None:
            raise AttributeError(self._VOCAB_ERROR_MSG)
        return len(self.vocab)

    def add(self, element: Hashable):
        if self.vocab is None:
            raise AttributeError(self._VOCAB_ERROR_MSG)
        return self.vocab.add_element(element)

    def has_element(self, element: Hashable) -> bool:
        if self.vocab is None:
            raise AttributeError(self._VOCAB_ERROR_MSG)
        return self.vocab.has_element(element)

    def element2repr(self, element: Hashable) -> Union[int, List[int]]:
        if self.vocab is None:
            raise AttributeError(self._VOCAB_ERROR_MSG)
        return self.vocab.element2repr(element)

    def id2element(self, idx: int) -> Any:
        if self.vocab is None:
            raise AttributeError(self._VOCAB_ERROR_MSG)
        return self.vocab.id2element(idx)

    def get_dict(self) -> Dict[Hashable, int]:
        if self.vocab is None:
            raise AttributeError(self._VOCAB_ERROR_MSG)
        return self.vocab.get_dict()

    def predefined_vocab(self, predefined: Union[Set, List]):
        r"""Functionality: Add elements from prediction into the vocabulary.

        Overwrite instruction:
            1. Take out elements from predefined.
            2. Make modification on elements, according to different
                Extractors.
            3. Use `self.add` function to add the element into vocabulary.

        Args:
            predefined (Union[Set, List]): A set or list contain
                elements to be added into the vocabulary.
        """
        for element in predefined:
            self.add(element)

    def update_vocab(self, pack: DataPack,
                     instance: Annotation):
        r"""Functionality: Add all elements from one instance into the
        vocabulary. For example, when the instance is Sentence and we want
        to add all Token from one sentence into the vocabulary, we might
        call this function.

        Overwrite instructions:
            1. Get all entries from one instance in the pack.
                You probably would use pack.get function to acquire
                Entry that you need.
            2. Get elements that are needed from entries. This process will
                be very different for different extractors. For example,
                you might want to get the token text from one sentence.
                Or you might want to get the tags for a sequence for
                one sentence.
            3. Use self.add to add those element into the vocabulary.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will get elements from.
        """
        pass

    def extract(self, pack: DataPack,
                instance: Annotation) -> Feature:
        r"""Functionality: Extract the feature for one instance in a pack.

        Overwrite instruction:
            1. Get all entries from one instance in the pack.
            2. Get elements that are needed form entries. For example,
                the token text or sequence tags.
            3. Construct a feature and return it.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will extractor feature.

        Returns:
            Feature: a feature that contains the extracted data.
        """
        pass

    def pre_evaluation_action(self, pack: DataPack,
                instance: Annotation):
        r"""This function is performed on the pack before the evaluation
        stage, allowing one to perform some actions before the evaluation.
        For example, you can remove entries or remove some attributes of
        the entry. By default, this function will not do anything.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance from which the
                extractor will extractor feature.
        """
        pass

    def add_to_pack(self, pack: DataPack, instance: Annotation,
                    prediction: Any):
        r"""Functionality: Add prediction to the pack.

        Overwrite instruction:
            1. Get all entries from one instance in the pack.
            2. Convert prediction into elements that need to be
                assigned to entries. You might need to use
                self.id2element to convert index in the prediction
                into element via the vocabulary maintained by the
                extractor.
            3. Use setattr to add the element to corresponding entry.

        Args:
            pack (Datapack): The datapack that contains the current
                instance.
            instance (Annotation): The instance to which the
                extractor add prediction.
            prediction (Any): This is the output of the model, whose
                format will be determined by the predict function
                user define and pass in to our framework.
        """
        pass
