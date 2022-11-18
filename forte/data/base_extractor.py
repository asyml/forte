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
import logging
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from typing import Union, Hashable, Iterable, Optional

from forte.common.configuration import Config
from forte.data.converter.feature import Feature
from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

__all__ = ["BaseExtractor"]


class BaseExtractor(ABC):
    r"""The functionality of Extractor is as followed. Most of
    the time, a user will not need to call this class explicitly,
    they will be called by the framework.

        1. Build vocabulary.
        2. Extract feature from datapack.
        3. Perform pre-evaluation action on datapack.
        4. Add prediction to datapack.

    Explanation:

        - Vocabulary:
          Vocabulary is maintained as an attribute
          in extractor. It will store the mapping from element
          to index, which is an integer, and representation,
          which could be an index integer or one-hot vector
          depending on the configuration of the vocabulary.
          Check :class:`~forte.data.vocabulary.Vocabulary` for
          more details.
        - Feature:
          A feature basically wraps the data we want from
          one instance in a datapack. For example, the instance
          can be one sentence in a datapack. Then the data wrapped
          by the feature could be the token text of this sentence.
          The data is already converted as list of indexes using
          vocabulary. Besides the data, other information like the
          raw data before indexing and some meta_data will also be
          stored in the feature. Check
          :class:`~forte.data.converter.Feature` for more
          details.
        - Remove feature / Add prediction:
          Removing feature means remove
          the existing data in the datapack. If we remove the feature
          in the pack, then extracting feature will return empty list.
          Adding prediction means we add the prediction from model
          back to the datapack. If a datapack has some old data (for
          example, the golden data in the test set), we can first
          remove those data and then add our model prediction to
          the pack.

    Attributes:
        config: An instance of `Dict` or :class:`~forte.common.configuration.Config` that
            provides configurable options. See
            :meth:`~forte.data.base_extractor.BaseExtractor.default_configs`
            for available options and default values.

    """
    _VOCAB_ERROR_MSG = (
        "When vocab_method is raw, vocabulary "
        "will not be built. Functions operating "
        "on vocabulary should not be called."
    )

    def __init__(self):
        self._vocab: Optional[Vocabulary] = None
        self.config: Config = None
        self._vocab_method = None

    def initialize(self, config: Union[Dict, Config]):
        self.config = Config(config, self.default_configs())

        if self.config.vocab_method != "custom":
            self._vocab = Vocabulary(
                method=self.config.vocab_method,
                use_pad=self.config.need_pad,
                use_unk=self.config.vocab_use_unk,
                pad_value=self.config.pad_value,
                unk_value=self.config.unk_value,
            )
        else:
            self._vocab = None
        self._vocab_method = self.config.vocab_method

    @classmethod
    def default_configs(cls):
        r"""Returns a dictionary of default hyper-parameters.

        Here:

        - vocab_method (str)
          What type of vocabulary is used for this extractor. `custom`,
          `indexing`, `one-hot` are supported, default is `indexing`.
          Check the behavior of vocabulary under different setting
          in :class:`~forte.data.vocabulary.Vocabulary`

        - context_type (str): The fully qualified name of the context used to
          group the extracted features, for example, it could be a
          `ft.onto.base_ontology.Sentence`. If this is `None`, features from
          in the whole data pack will be grouped together. Default is None.
          This value could be mandatory for some processors, which will be
          documented and specified by the specific processor implementation.

        - vocab_use_unk (bool)
          Whether the `<UNK>` element should be added to vocabulary.
          Default is true.

        - need_pad (bool)
          Whether the `<PAD>` element should be added to vocabulary. And
          whether the feature need to be batched and padded. Default is True.

        - pad_value (int)
          A customized value/representation to be used for
          padding. This value is only needed when `use_pad` is True.
          Default is None, where the value of padding is determined by
          the system.

        - unk_value (int)
          A customized value/representation to be used for
          unknown value (`unk`). This value is only needed when
          `vocab_use_unk` is True. Default is None, where the value
          of `UNK` is determined by the system.

        """
        return {
            "vocab_method": "indexing",
            "context_type": None,
            "vocab_use_unk": True,
            "need_pad": True,
            "pad_value": None,
            "unk_value": None,
        }

    @property
    def vocab_method(self) -> str:
        return self._vocab_method

    @property
    def vocab(self) -> Optional[Vocabulary]:
        """
        Getter of the vocabulary class.

        Returns: The vocabulary. None if the vocabulary is not set.

        """
        return self._vocab

    @vocab.setter
    def vocab(self, vocab: Vocabulary):
        """
        Setter of the vocabulary, used when user build the vocabulary
        externally.

        Args:
            vocab: The vocabulary to be assigned.

        Returns:

        """
        self._vocab = vocab

    def get_pad_value(self) -> Union[None, int, List[int]]:
        if self.vocab is not None:
            return self.vocab.get_pad_value()
        else:
            return None

    def vocab_items(self) -> Iterable[Tuple[Hashable, int]]:
        if self.vocab is None:
            raise AttributeError(self._VOCAB_ERROR_MSG)
        return self.vocab.vocab_items()

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

    def predefined_vocab(self, predefined: Iterable):
        r"""Populate the vocabulary with predefined values. You can also extend
        this method to customize the ways to handle the vocabulary.

        Overwrite instruction:

            1. Take out elements from predefined.
            2. Make modification on the elements based on the need of the
               extractor.
            3. Use :meth:`add` function to add the element into vocabulary.

        Args:
            predefined (Iterable): A collections that contains the elements to
              be added into the vocabulary.
        """
        for element in predefined:
            self.add(element)

    def update_vocab(
        self, pack: DataPack, context: Optional[Annotation] = None
    ):
        r"""Populate the vocabulary needed by the extractor. This can be
        implemented by a specific extractor. The populated vocabulary can be
        used to map features/items to numeric representations. If you use a
        pre-specified vocabulary, you may not need to use this function.

        Overwrite instructions:

            1. Get all entries of the type of interest, such as all the
            `Token`s in the data pack.
            2. Use :meth:`~forte.data.vocabulary.Vocabulary.add` to add those
            element into `self._vocab`.

        Args:
            pack: The input data pack.
            context: The context is an Annotation entry where
                features will be extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.
        """
        pass

    @abstractmethod
    def extract(
        self, pack: DataPack, context: Optional[Annotation] = None
    ) -> Feature:
        """This method should be implemented to extract features from a
        datapack.

        Args:
            pack: The input data pack that contains the features.
            context: The context is an Annotation entry where
                features will be extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.

        Returns: Features inside this instance stored as a
        `~forte.data.converter.feature.Feature` instance.

        """
        raise NotImplementedError

    def pre_evaluation_action(
        self, pack: DataPack, context: Optional[Annotation]
    ):
        r"""This function is performed on the pack before the evaluation
        stage, allowing one to perform some actions before the evaluation.
        For example, you can remove entries or remove some attributes of
        the entry. By default, this function will not do anything.

        Args:
            pack: The datapack that contains the current instance.
            context: The context is an Annotation entry
                where data are extracted within its range. If None, then the
                whole data pack will be used as the context. Default is None.
        """
        pass

    def add_to_pack(
        self,
        pack: DataPack,
        predictions: Any,
        context: Optional[Annotation] = None,
    ):
        r"""Add prediction of a model (normally in the form of a tensor)
        back to the pack. This function should have knowledge of the structure
        of the `prediction` to correctly populate the data pack values.

        This function can be roughly considered as the reverse operation of
        :meth:`~forte.data.base_extractor.BaseExtractor.extract`.

        Overwrite instruction:

            1. Get all entries from one instance in the pack.

            2. Convert predictions into elements that needs to be assigned
               to entries. You can use :meth:`~forte.data.vocabulary.id2element`
               to convert integers in the prediction into element via the
               vocabulary maintained by the extractor.

            3. Add the element to corresponding entry based on the need.

        Args:
            pack: The datapack to add predictions back.
            predictions: This is the output of the model, the format of
              which will be determined by the predict function defined in the
              Predictor.
            context: The context is an Annotation
                entry where predictions will be added to. This has the same
                meaning with `context` as in
                :meth:`~forte.data.base_extractor.BaseExtractor.extract`.
                If None, then the whole data pack will be used as the
                context. Default is None.
        """
        pass
