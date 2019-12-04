from forte.indexers import ElasticSearchIndexer
from forte.data.readers import StringReader
from forte.pipeline import Pipeline

from ft.onto.base_ontology import Document

if __name__ == "__main__":
    nlp = Pipeline()
    nlp.set_reader(StringReader())
    passages = []
    dataset = ["Passage1", "Passage2", "Passage3"]
    for pack in nlp.process_dataset(dataset):
        texts = [p.text for p in pack.get_entries_by_type(Document)]
        passages.extend(texts)

    es = ElasticSearchIndexer()
    documents = [{"content": passage} for passage in passages]
    es.add_bulk(documents)
