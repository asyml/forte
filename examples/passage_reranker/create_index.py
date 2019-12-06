import texar.torch as tx
from forte.indexers import ElasticSearchIndexer
from forte.data.readers import CorpusReader
from forte.processors import ElasticSearchIndexProcessor
from forte.pipeline import Pipeline

if __name__ == "__main__":
    nlp = Pipeline()
    nlp.set_reader(CorpusReader())
    config = tx.HParams({
        "batch_size": 1000000,
        "field": "content",
        "indexer": {
            "name": "ElasticSearchIndexer",
            "hparams": ElasticSearchIndexer.default_hparams(),
            "other_kwargs": {
                "request_timeout": 60,
                "refresh": False
            }
        }
    }, default_hparams=None)
    nlp.add_processor(ElasticSearchIndexProcessor(), config=config)
    nlp.initialize()

    count = 0
    for pack in nlp.process_dataset("."):
        if count > 0 and count % 10000 == 0:
            print(f"Completed {count} packs")
        count += 1
