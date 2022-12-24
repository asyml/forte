#  Copyright 2020 The Forte Authors. All Rights Reserved.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#       http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
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
Provide a framework inherited by users to do the training.
"""
import logging
from abc import abstractmethod
from typing import Dict, Iterator, Any, Optional
import pickle

from forte.data.data_pack import DataPack
from forte.train_preprocessor import TrainPreprocessor

logger = logging.getLogger(__name__)

__all__ = ["BaseTrainer"]


class BaseTrainer:
    r"""
    `BaseTrainer` is the main entry for using Forte training framework. Users
    should inherit this class and overwrite multiple methods defined in this
    class. Internally, it will make use of
    :class:`~forte.train_preprocessor.TrainPreprocessor` to do the the actual
    training. Please refer to the documentation of that class for details. A
    concrete example is provided below showing how to use this class.

    Below is an example for how to use this class. A fully documented example is
    also provided in
    :class:`~forte.examples.tagging.tagging_trainer.TaggingTrainer`.

    .. code-block:: python

        class TaggingTrainer(BaseTrainer):
            def create_tp_config(self) -> Dict:
                tp_request: Dict = {
                    "scope": Sentence,
                    "schemes": {
                        "text_tag": {
                            "type": TrainPreprocessor.DATA_INPUT,
                            "extractor": {
                                "class_name":
                                "forte.data.extractors.AttributeExtractor",
                                "config": {
                                    "entry_type": "ft.onto.base_ontology.Token",
                                    "vocab_method": "indexing",
                                    "attribute": "text"
                                }
                            }
                        },
                        "char_tag": {
                            "type": TrainPreprocessor.DATA_INPUT,
                            "extractor": {
                                "class_name":
                                "forte.data.extractors.CharExtractor",
                                "config": {
                                    "entry_type": "ft.onto.base_ontology.Token",
                                    "vocab_method": "indexing",
                                    "max_char_length": 45
                                }
                            }
                        },
                        "output_tag": {
                            "type": TrainPreprocessor.DATA_OUTPUT,
                            "extractor": {
                                "class_name":
                                "forte.data.extractors.BioSeqTaggingExtractor",
                                "config": {
                                    "vocab_method": "indexing"
                                    "pad_value": 0
                                }
                            }
                        }
                    }
                }
                return tp_request

            def create_pack_iterator(self) -> Iterator[DataPack]:
                reader = CoNLL03Reader()
                train_pl: Pipeline = Pipeline()
                train_pl.set_reader(reader)
                train_pl.initialize()
                pack_iterator: Iterator[DataPack] = \
                    train_pl.process_dataset(self.config_data.train_path)
                return pack_iterator

            def train(self):
                schemes: Dict = self.train_preprocessor.request["schemes"]
                text_extractor: BaseExtractor = \
                    schemes["text_tag"]["extractor"]
                char_extractor: BaseExtractor = \
                    schemes["char_tag"]["extractor"]
                output_extractor: BaseExtractor = \
                    schemes["output_tag"]["extractor"]

                model: BiRecurrentConvCRF = \
                    BiRecurrentConvCRF(word_vocab=text_extractor.get_dict(),
                                       char_vocab_size=char_extractor.size(),
                                       tag_vocab_size=output_extractor.size(),
                                       config_model=self.config_model)
                model.to(self.device)

                optim: Optimizer = SGD(model.parameters())

                tp = self.train_preprocessor

                # Train for 10 epochs
                for epoch in range(10):
                    # Get iterator of preprocessed batch of train data
                    batch_iter: Iterator[Batch] = tp.get_train_batch_iterator()

                    for batch in tqdm(batch_iter):
                        word = batch["text_tag"]["data"]
                        char = batch["char_tag"]["data"]
                        output = batch["output_tag"]["data"]
                        word_masks = batch["text_tag"]["masks"][0]

                        optim.zero_grad()

                        loss = model(word, char, output, mask=word_masks)

                        loss.backward()
                        optim.step()
    """

    def __init__(self):
        self._pack_iterator: Optional[Iterator[DataPack]] = None
        self._tp: Optional[TrainPreprocessor] = None
        self._initialized: bool = False

    def initialize(self):
        # Check if initialize has already been called before
        if self._initialized:
            return

        logging.info("Initializing the trainer...")
        self._pack_iterator = self.create_pack_iterator()
        self._tp = TrainPreprocessor(pack_iterator=self._pack_iterator)
        self._tp.initialize(config=self.create_tp_config())
        self._initialized = True
        logging.info("Done Initializing.")

    @property
    def train_preprocessor(self) -> Optional[TrainPreprocessor]:
        r"""The instance of type
        :class:`~forte.train_preprocessor.TrainPreprocessor`. The Trainer will
        internally create an instance of this class to do the actual training.
        """
        if not self._initialized:
            raise ValueError(
                "initialize should be called to build train preprocessor."
            )
        return self._tp

    def run(self):
        r"""The entry point for starting a training process."""
        self.initialize()
        self.train()

    @abstractmethod
    def create_tp_config(self) -> Dict:
        r"""Users should overwrite this method to provide a concrete train
        preprocessor config to create an object of
        :class:`~forte.train_preprocessor.TrainPreprocessor`.
        An example config is given in the example above.

        Please refer to :meth:`default_configs` in class
        :class:`~forte.train_preprocessor.TrainPreprocessor` for detailed
        specification of each options in the config.
        """
        raise NotImplementedError

    @abstractmethod
    def create_pack_iterator(self) -> Iterator[DataPack]:
        r"""Users should overwrite this method to provide an iterator of data,
        in DataPack formats:class:`~forte.data.data_pack.DataPack`.
        This iterator will be used to produce each input data pack consumed
        for training. Typically, users can create a reader of type
        :class:`~forte.data.readers.base_reader.BaseReader`. The reader can be
        wrapped as an iterator of data pack via forte pipeline system. Please
        refer to the above example for how to create this.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self):
        r"""Users should overwrite this method to provide the detail logic of
        doing the training (forward and backward processing). Users can use the
        :meth:`get_train_batch_iterator` in class
        :class:`~forte.train_preprocessor.TrainPreprocessor` to get an iterator
        of pre-processed batch of data. Please refer to that method for details.
        An example is also provided above.
        """
        raise NotImplementedError

    def save(self, *args: Any, **kwargs: Any):
        r"""Save the training states to disk for the usage of later
        predicting phase. The default training states is the request inside
        TrainPreprocessor. Please refer to :meth:`request` in class
        :class:`~forte.train_preprocessor.TrainPreprocessor` for details.
        Typically users do not need to overwrite this method as default saved
        training state is enough for predicting usage. But users can also
        overwrite this method to achieve special purpose.
        """

        # Check arg type. Default behavior only supports str as args[0] which
        # is considered as a disk file path.
        if not isinstance(args[0], str):
            raise ValueError(
                f"Do not support input args: {args} and kwargs: {kwargs}"
            )

        file_path = args[0]

        if not isinstance(self.train_preprocessor, TrainPreprocessor):
            raise ValueError(
                f"Invalid TrainPreprocessor type: {self.train_preprocessor}"
            )

        request: Dict = self.train_preprocessor.request

        with open(file_path, "wb") as f:
            pickle.dump(request, f)
