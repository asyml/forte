import logging
from typing import Dict, Any, Optional

from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.indexers import ElasticSearchIndexer
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Utterance


# pylint: disable=attribute-defined-outside-init


def new_utterance(input_pack: DataPack, text: str, speaker: str):
    input_pack.set_text(input_pack.text + '\n' + text)
    logging.info('The response is:')
    logging.info(text)

    u = Utterance(input_pack,
                  len(input_pack.text) - len(text),
                  len(input_pack.text))
    u.speaker = speaker


class LastUtteranceSearcher(PackProcessor):
    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.index = ElasticSearchIndexer(self.configs.indexer.hparams)

    def _process(self, input_pack: DataPack):
        # Make sure we take the last utterance from the user.
        utterance: Optional[Utterance] = None
        u: Utterance
        for u in input_pack.get(Utterance):
            if u.speaker == 'user':
                utterance = u

        if utterance:
            logging.info("The last utterance is %s", utterance)
        else:
            logging.info("Cannot get another utterance.")
            new_utterance(
                input_pack,
                "Hey, I didn't get what you say, could you try again?",
                'ai')

        # Create the query using the last utterance from user.
        size = self.configs.size or 1000
        field = self.configs.field or "content"
        query_value = {"query": {"match": {field: utterance.text}},
                       "size": size}

        # Search against the index.
        results = self.index.search(query_value)
        hits = results["hits"]["hits"]

        for idx, hit in enumerate(hits):
            source = hit["_source"]
            # The raw pack string and pack id (not database id)
            raw_pack_str: str = source["pack_info"]
            pack_id: str = source["doc_id"]

            # Now you can write the pack into the database, and generate url.
            print(pack_id)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update({
            "size": 5,
            "field": "content",
            "indexer": {
                "name": "ElasticSearchIndexer",
                "hparams": ElasticSearchIndexer.default_configs(),
                "other_kwargs": {
                    "request_timeout": 10,
                    "refresh": False
                }
            }
        })
        return config
