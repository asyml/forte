import yaml
import os

import texar.torch as tx

from forte.processors import (ElasticSearchQueryCreator, ElasticSearchProcessor)
from forte.pipeline import Pipeline

from reader import EvalReader
from ms_marco_evaluator import MSMarcoEvaluator
from reranking_processor import RerankingProcessor

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yml", "r"))
    config = tx.HParams(config, default_hparams=None)
    ms_marco_evaluator = MSMarcoEvaluator()

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                             "collectionandqueries")
    query_path = os.path.join(data_path, "queries.dev.small.tsv")

    nlp = Pipeline()
    nlp.set_reader(reader=EvalReader(), config=config.reader)
    nlp.add_processor(processor=ElasticSearchQueryCreator(),
                      config=config.query_creator)
    nlp.add_processor(processor=ElasticSearchProcessor(), config=config.indexer)
    nlp.add_processor(processor=RerankingProcessor(), config=config.reranker)

    nlp.set_evaluator(evaluator=ms_marco_evaluator, config=config.evaluator)
    nlp.initialize()

    for idx, m_pack in enumerate(nlp.process_dataset(query_path)):
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx+1} examples")

    scores = nlp.evaluate()
    print(scores)
