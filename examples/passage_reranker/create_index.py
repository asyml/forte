import texar.torch as tx

from forte.data.readers import CorpusReader
from forte.processors import ElasticSearchIndexProcessor
from forte.pipeline import Pipeline

if __name__ == "__main__":
    nlp = Pipeline()
    nlp.set_reader(CorpusReader())
    config = tx.HParams({
        "batch_size": 100000,
        "fields": ["doc_id", "content"],
        "indexer": {
            "name": "ElasticSearchIndexer",
            "hparams": {
                "index_name": "elastic_indexer2",
                "hosts": "localhost:9200",
                "algorithm": "bm25"
            },
            "other_kwargs": {
                "request_timeout": 60,
                "refresh": False
            }
        }
    }, default_hparams=None)
    nlp.add_processor(ElasticSearchIndexProcessor(), config=config)
    nlp.initialize()

    for idx, pack in enumerate(nlp.process_dataset(".")):
        if idx + 1 > 0 and (idx + 1) % 100000 == 0:
            print(f"Completed {idx+1} packs")
