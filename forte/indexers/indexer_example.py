import os

from termcolor import colored
import torch
from texar.torch.modules import BERTEncoder
from texar.torch import HParams
from texar.torch.data import RecordData, DataIterator

from forte.indexers import EmbeddingBasedIndexer
from forte.data.ontology import base_ontology
from forte.data.ontology import ontonotes_ontology as onto
from forte.data.readers import StringReader, SimpleSearcher
from forte.pipeline import Pipeline
from forte.processors import (NLTKSentenceSegmenter, NLTKWordTokenizer,
                              NLTKPOSTagger, SRLPredictor, QueryCreator)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_seq_length = 512
batch_size = 128
pickle_data_dir = "forte/indexers/data/imdb2.pkl"
model_dir = "index.mdl"
srl_model_dir = "models/srl"
vocab_file = "forte/indexers/pretrained_models/uncased_L-12_H-768_A-12/vocab.txt"


@torch.no_grad()
def get_embeddings(self, input_ids, segment_ids):
    return self.encoder(inputs=input_ids, segment_ids=segment_ids)


def main():

    if not os.path.exists(model_dir):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = BERTEncoder(pretrained_model_name="bert-base-uncased")
        encoder.to(device)

        feature_original_types = {
            "id": ["int64", "FixedLenFeature"],
            "input_ids": ["int64", "FixedLenFeature", max_seq_length],
            "segment_ids": ["int64", "FixedLenFeature", max_seq_length],
            "text": ["str", "FixedLenFeature"]
        }

        hparam = {
            "allow_smaller_final_batch": True,
            "batch_size": batch_size,
            "dataset": {
                "data_name": "data",
                "feature_original_types": feature_original_types,
                "files": pickle_data_dir
            },
            "shuffle": False
        }

        print(f"Embedding the text using BERTEncoder...")
        record_data = RecordData(hparams=hparam, device=device)
        data_iterator = DataIterator(record_data)
        index = EmbeddingBasedIndexer(hparams={
            "index_type": "GpuIndexFlatIP",
            "dim": 768
        })

        for idx, batch in enumerate(data_iterator):
            ids = batch["id"]
            input_ids = batch["input_ids"]
            segment_ids = batch["segment_ids"]
            text = batch["text"]
            _, pooled_output = get_embeddings(input_ids, segment_ids)
            index.add(vectors=pooled_output,
                      meta_data={k.item(): v for k,v in zip(ids, text)})

            if (idx + 1) % 50 == 0:
                print(f"Completed {idx+1} batches of size {batch_size}")

        index.save(path=model_dir)

    queries = ["romantic comedy movies"]

    query_pipeline = Pipeline()
    query_pipeline.set_reader(StringReader())
    query_processor_config = HParams(
        {
            "index": model_dir,
            "vocab_file": "forte/indexers/pretrained_models/uncased_L-12_H-768_A-12/vocab.txt"
        },
        default_hparams=None
    )
    query_pipeline.add_processor(QueryCreator(), query_processor_config)
    query_pipeline.initialize_processors()

    # search pipeline
    search_pipeline = Pipeline()
    search_pipeline.set_reader(SimpleSearcher(index_path=model_dir))
    search_pipeline.add_processor(NLTKSentenceSegmenter())
    search_pipeline.add_processor(NLTKWordTokenizer())
    search_pipeline.add_processor(NLTKPOSTagger())

    srl_configs = HParams(
        {
            'storage_path': srl_model_dir,
        },
        SRLPredictor.default_hparams()
    )
    search_pipeline.add_processor(SRLPredictor(), srl_configs)
    search_pipeline.initialize_processors()

    for pack in query_pipeline.process_dataset(queries):
        processed_pack = search_pipeline.process_one(pack)
        for sentence in processed_pack.get(base_ontology.Sentence):
            sent_text = sentence.text
            print(colored("Sentence:", 'red'), sent_text, "\n")

            # first method to get entry in a sentence
            tokens = [(token.text, token.pos_tag) for token in
                      processed_pack.get(base_ontology.Token, sentence)]
            print(colored("Tokens:", 'red'), tokens, "\n")

            #entities = [(entity.text, entity.ner_type) for entity in
            #            processed_pack.get(base_ontology.EntityMention, sentence)]
            #print(colored("EntityMentions:", 'red'), entities, "\n")

            print(colored(sentence.text, "green"))

            # second method to get entry in a sentence
            print(colored("Semantic role labels:", 'red'))
            for link in processed_pack.get(onto.PredicateLink, sentence):
                parent = link.get_parent()
                child = link.get_child()
                print(f"  - \"{child.text}\" is role {link.arg_type} of "
                      f"predicate \"{parent.text}\"")
                entities = [entity.text for entity
                            in processed_pack.get(onto.EntityMention, child)]
                print("Entities in predicate argument:", entities, "\n")


if __name__ == "__main__":
    main()
