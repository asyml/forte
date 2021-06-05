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

import os


import yaml
from forte.common.configuration import Config
from forte.data.caster import MultiPackBoxer
from forte.data.readers import TerminalReader
from forte.data.multi_pack import MultiPack

from forte.pipeline import Pipeline
from forte_wrapper.vader import VaderSentimentProcessor
from forte_wrapper.twitter import TweetSearchProcessor
from forte.data.selector import RegexNameMatchSelector


if __name__ == "__main__":
    # Load config file
    config_file = os.path.join(os.path.dirname(__file__), 'config.yml')
    config = yaml.safe_load(open(config_file, "r"))
    config = Config(config, default_hparams=None)

    # Build pipeline and add the reader, which will read query from terminal.
    nlp: Pipeline = Pipeline()
    nlp.set_reader(reader=TerminalReader())

    # Start to work on multi-packs in the rest of the pipeline, so we use a
    # boxer to change this.
    nlp.add(MultiPackBoxer(), config=config.boxer)

    # Search tweets.
    nlp.add(TweetSearchProcessor(), config=config.twitter_search)

    # Conduct sentiment analysis.
    pattern = rf"{config.twitter_search.response_pack_name_prefix}_\d"
    selector_hit = RegexNameMatchSelector(select_name=pattern)
    nlp.add(component=VaderSentimentProcessor(),
            selector=selector_hit, config=config.vader_sentiment)

    nlp.initialize()

    # process dataset
    m_pack: MultiPack
    for m_pack in nlp.process_dataset():
        print('The number of datapacks(including query) is', len(m_pack.packs))

        for pack in m_pack.packs:
            # Do not process the query datapack
            if pack.pack_name == config.twitter_search.query_pack_name:
                continue
            for sentence in pack.get(config.vader_sentiment.entry_type):
                print('Tweet: ', sentence.text)
                print('Sentiment Score: ', sentence.sentiment['compound'])

    print('Done')