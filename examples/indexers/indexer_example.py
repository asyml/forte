import os
import yaml

from termcolor import colored
import torch
from texar.torch.modules import BERTEncoder
from texar.torch import HParams
from texar.torch.data import RecordData, DataIterator

from forte.indexers import EmbeddingBasedIndexer
# from forte.data.ontology import conll03_ontology as conll
from forte.data.ontology import base_ontology as ontology
from forte.data.readers import MultiPackTerminalReader
from forte.common.resources import Resources
from forte.pipeline import Pipeline
from forte.processors import SRLPredictor
from forte.processors.machine_translation_processor import \
    MachineTranslationProcessor
from forte.processors.query_creator import QueryCreator
from forte.processors.search_processor import SearchProcessor
from forte.data.selector import NameMatchSelector
from examples.processors.NLTK_processors import \
    (NLTKSentenceSegmenter, NLTKWordTokenizer, NLTKPOSTagger)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def get_embeddings(encoder, input_ids, segment_ids):
    return encoder(inputs=input_ids, segment_ids=segment_ids)


def main():

    config = yaml.safe_load(open("config.yml", "r"))
    config = HParams(config, default_hparams=None)

    if not os.path.exists(config.indexer.model_dir):
        print(f"Creating a new index...")
        encoder = BERTEncoder(pretrained_model_name="bert-base-uncased")
        encoder.to(device)

        feature_original_types = {
            "id": ["int64", "FixedLenFeature"],
            "input_ids": ["int64", "FixedLenFeature",
                          config.indexer.max_seq_length],
            "segment_ids": ["int64", "FixedLenFeature",
                            config.indexer.max_seq_length],
            "text": ["str", "FixedLenFeature"]
        }

        hparam = {
            "allow_smaller_final_batch": True,
            "batch_size": config.indexer.batch_size,
            "dataset": {
                "data_name": "data",
                "feature_original_types": feature_original_types,
                "files": config.indexer.pickle_data_dir
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
            _, pooled_output = get_embeddings(encoder, input_ids, segment_ids)
            index.add(vectors=pooled_output,
                      meta_data={k.item(): v for k, v in zip(ids, text)})

            if (idx + 1) % 50 == 0:
                print(f"Completed {idx+1} batches of size "
                      f"{config.indexer.batch_size}")

        index.save(path=config.indexer.model_dir)

    resource = Resources()
    query_pipeline = Pipeline(resource=resource)
    query_pipeline.set_reader(MultiPackTerminalReader())

    query_pipeline.add_processor(
        processor=MachineTranslationProcessor(), config=config.translator)
    query_pipeline.add_processor(
        processor=QueryCreator(), config=config.query_creator)
    query_pipeline.add_processor(
        processor=SearchProcessor(), config=config.indexer)
    query_pipeline.add_processor(
        processor=NLTKSentenceSegmenter(),
        selector=NameMatchSelector(select_name="doc_0"))
    query_pipeline.add_processor(
        processor=NLTKWordTokenizer(),
        selector=NameMatchSelector(select_name="doc_0"))
    query_pipeline.add_processor(
        processor=NLTKPOSTagger(),
        selector=NameMatchSelector(select_name="doc_0"))
    query_pipeline.add_processor(
        processor=SRLPredictor(), config=config.SRL,
        selector=NameMatchSelector(select_name="doc_0"))
    # query_pipeline.add_processor(
    #    processor=CoNLLNERPredictor(), config=config.NER,
    #    selector=NameMatchSelector(select_name="doc_0"))
    query_pipeline.add_processor(
        processor=MachineTranslationProcessor(), config=config.back_translator)

    query_pipeline.initialize()
    query_pipeline.set_ontology(ontology)

    for m_pack in query_pipeline.process_dataset():

        # update resource to be used in the next conversation
        query_pack = m_pack.get_pack("query")
        if resource.get("user_utterance"):
            resource.get("user_utterance").append(query_pack)
        else:
            resource.update(user_utterance=[query_pack])

        response_pack = m_pack.get_pack("response")

        if resource.get("bot_utterance"):
            resource.get("bot_utterance").append(response_pack)
        else:
            resource.update(bot_utterance=[response_pack])

        english_pack = m_pack.get_pack("pack")
        print(colored("English Translation of the query: ", "green"),
              english_pack.text, "\n")
        pack = m_pack.get_pack("doc_0")
        print(colored("Retrieved Document", "green"), pack.text, "\n")
        print(colored("German Translation", "green"),
              m_pack.get_pack("response").text, "\n")
        for sentence in pack.get(ontology.Sentence):
            sent_text = sentence.text
            print(colored("base_ontology.Sentence:", 'red'), sent_text, "\n")

            print(colored("Semantic role labels:", 'red'))
            for link in pack.get(ontology.PredicateLink, sentence):
                parent = link.get_parent()
                child = link.get_child()
                print(f"  - \"{child.text}\" is role {link.arg_type} of "
                      f"predicate \"{parent.text}\"")
            print()

            input(colored("Press ENTER to continue...\n", 'green'))


if __name__ == "__main__":
    main()
