import yaml

import os
from termcolor import colored

import texar.torch as tx

from forte.data.selector import RegexNameMatchSelector
from forte.data.readers import MultiPackTerminalReader
from forte.processors import (
    ElasticSearchQueryCreator, ElasticSearchProcessor,
    BERTBasedRerankingProcessor, NLTKWordTokenizer, NLTKPOSTagger,
    NLTKSentenceSegmenter, CoNLLNERPredictor)
from forte.pipeline import Pipeline

from ft.onto.base_ontology import Sentence, Token, EntityMention


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), 'config.yml')
    config = yaml.safe_load(open(config_file, "r"))
    config = tx.HParams(config, default_hparams=None)

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "collectionandqueries")
    query_path = os.path.join(data_path, "queries.dev.small.tsv")
    doc_pack_name = '.[0-9]*'

    nlp = Pipeline()
    nlp.set_reader(reader=MultiPackTerminalReader(), config=config.reader)

    # Indexing and Re-ranking
    nlp.add_processor(processor=ElasticSearchQueryCreator(),
                      config=config.query_creator)
    nlp.add_processor(processor=ElasticSearchProcessor(),
                      config=config.indexer)
    nlp.add_processor(processor=BERTBasedRerankingProcessor(),
                      config=config.reranker)

    # NER Tagging
    nlp.add_processor(processor=NLTKSentenceSegmenter(),
                      selector=RegexNameMatchSelector(doc_pack_name))
    nlp.add_processor(processor=NLTKWordTokenizer(),
                      selector=RegexNameMatchSelector(doc_pack_name))
    nlp.add_processor(processor=NLTKPOSTagger(),
                      selector=RegexNameMatchSelector(doc_pack_name))
    nlp.add_processor(CoNLLNERPredictor(), config=config.NER,
                      selector=RegexNameMatchSelector(doc_pack_name))

    nlp.initialize()

    passages = [f"passage_{i}" for i in range(config.query_creator.size)]
    
    for idx, m_pack in enumerate(nlp.process_dataset()):
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx+1} examples")
        for passage in passages:
            pack = m_pack.get_pack(passage)
            print(colored("Passage: ", "green"), pack.text, "\n")
            for sentence in pack.get(Sentence):
                sent_text = sentence.text
                print(colored("Sentence:", 'green'), sent_text, "\n")
                # first method to get entry in a sentence
                tokens = [(token.text, token.pos) for token in
                          pack.get(Token, sentence)]
                entities = [(entity.text, entity.ner_type) for entity in
                            pack.get(EntityMention, sentence)]
                print(colored("Tokens:", 'red'), tokens, "\n")
                print(colored("EntityMentions:", 'red'), entities, "\n")
                input(colored("Press ENTER to continue...\n", 'green'))
