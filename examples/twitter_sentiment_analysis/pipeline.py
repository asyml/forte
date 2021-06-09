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
from forte.vader import VaderSentimentProcessor
from forte.tweepy import TweetSearchProcessor
from forte.data.selector import RegexNameMatchSelector


if __name__ == "__main__":
    # Load config file
    config_file = os.path.join(os.path.dirname(__file__), "config.yml")
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
    nlp.add(
        component=VaderSentimentProcessor(),
        selector=selector_hit,
        config=config.vader_sentiment,
    )

    nlp.initialize()

    # process dataset
    m_pack: MultiPack
    for m_pack in nlp.process_dataset():
        print("The number of datapacks(including query) is", len(m_pack.packs))

        tweets, pos_sentiment, neg_sentiment, neutral_sentiment = 0, 0, 0, 0

        for name, pack in m_pack.iter_packs():
            # Do not process the query datapack
            if name == config.twitter_search.query_pack_name:
                continue

            tweets += 1
            for doc in pack.get(config.vader_sentiment.entry_type):
                print("Tweet: ", doc.text)
                print("Sentiment Compound Score: ", doc.sentiment["compound"])

                compound_score = doc.sentiment["compound"]
                if compound_score >= 0.05:
                    pos_sentiment += 1
                elif compound_score <= -0.05:
                    neg_sentiment += 1
                else:
                    neutral_sentiment += 1

        print("The number of tweets retrieved: ", tweets)
        print("The proportion of positive sentiment: ", pos_sentiment / tweets)
        print("The proportion of negative sentiment: ", neg_sentiment / tweets)
        print(
            "The proportion of neutral sentiment: ", neutral_sentiment / tweets
        )

    print("Done")
