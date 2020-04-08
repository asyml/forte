# Copyright 2019 The Forte Authors. All Rights Reserved.
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

__all__ = [
    "VaderSentimentProcessor",
]

from texar.torch import HParams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from forte.common import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Sentence


class VaderSentimentProcessor(PackProcessor):
    r"""A wrapper of a sentiment analyzer: Vader (Valence Aware Dictionary
    and sEntiment Reasoner). Vader needs to be installed to use this package

     > pip install vaderSentiment
     or
     > pip install --upgrade vaderSentiment

    This processor will add assign sentiment label to each sentence in the
    document. If the input pack contains no sentence then no processing will
    happen. If the data pack has multiple set of sentences, one can specify
    the set of sentences to tag by setting the `sentence_component` attribute.

    Vader URL: (https://github.com/cjhutto/vaderSentiment)
    Citation: VADER: A Parsimonious Rule-based Model for Sentiment Analysis
       of Social Media Text (by C.J. Hutto and Eric Gilbert)

    """

    def __init__(self):
        super().__init__()
        self.sentence_component = None
        self.analyzer = SentimentIntensityAnalyzer()

    def initialize(self, resources: Resources, configs: HParams):
        # pylint: disable=unused-argument
        self.sentence_component = configs.get('sentence_component')

    def _process(self, input_pack: DataPack):
        for sentence in input_pack.get(entry_type=Sentence,
                                       component=self.sentence_component):
            scores = self.analyzer.polarity_scores(sentence.text)
            sentence.sentiment = scores

    @classmethod
    def default_configs(cls):
        r"""This defines a basic config structure for VaderSentimentProcessor.

        sentence_component (str): If not None, the processor will process
          sentence with the provided component name. If None, then all sentences
          will be processed.
        """
        config = super().default_configs()
        config.update({
            'sentence_component': None
        })
        return config
