import yaml

import texar.torch as tx
from forte.processors import ElasticSearchQueryCreator, ElasticSearchProcessor
from forte.pipeline import Pipeline

from reader import EvalReader
from ms_marco_evaluator import MSMarcoEvaluator

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yml", "r"))
    config = tx.HParams(config, default_hparams=None)
    ms_marco_evaluator = MSMarcoEvaluator()

    nlp = Pipeline()
    nlp.set_reader(reader=EvalReader(), config=config.reader)
    nlp.add_processor(processor=ElasticSearchQueryCreator(),
                      config=config.query_creator)
    nlp.add_processor(processor=ElasticSearchProcessor(), config=config.indexer)
    nlp.set_evaluator(evaluator=ms_marco_evaluator, config=config.evaluator)
    nlp.initialize()

    for idx, m_pack in enumerate(nlp.process_dataset(
            "./collection_and_queries/queries.dev.small.tsv")):
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx+1} examples")

    scores = nlp.evaluate()
    print(scores)
