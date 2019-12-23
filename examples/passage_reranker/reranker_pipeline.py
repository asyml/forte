import yaml
from termcolor import colored

import texar.torch as tx
from forte.data.selector import RegexNameMatchSelector
from forte.data.readers import MultiPackTerminalReader
from forte.processors import ElasticSearchQueryCreator, ElasticSearchProcessor
from forte.pipeline import Pipeline
from forte.processors import CoNLLNERPredictor
from forte.processors.nltk_processors import NLTKWordTokenizer, \
    NLTKPOSTagger, NLTKSentenceSegmenter

from ft.onto.base_ontology import Sentence, Token, EntityMention

from reader import EvalReader
from ms_marco_evaluator import MSMarcoEvaluator

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yml", "r"))
    config = tx.HParams(config, default_hparams=None)
    ms_marco_evaluator = MSMarcoEvaluator()

    nlp = Pipeline()
    # nlp.set_reader(reader=EvalReader(), config=config.reader)
    nlp.set_reader(reader=MultiPackTerminalReader(), config=config.reader)
    nlp.add_processor(processor=ElasticSearchQueryCreator(),
                      config=config.query_creator)
    nlp.add_processor(processor=ElasticSearchProcessor(), config=config.indexer)
    nlp.add_processor(processor=NLTKSentenceSegmenter(),
                      selector=RegexNameMatchSelector(select_name=config.indexer.response_pack_name_prefix))
    nlp.add_processor(processor=NLTKWordTokenizer(),
                      selector=RegexNameMatchSelector(select_name=config.indexer.response_pack_name_prefix))
    nlp.add_processor(processor=NLTKPOSTagger(),
                      selector=RegexNameMatchSelector(select_name=config.indexer.response_pack_name_prefix))
    nlp.add_processor(CoNLLNERPredictor(), config=config.NER,
                      selector=RegexNameMatchSelector(select_name=config.indexer.response_pack_name_prefix))
    # nlp.set_evaluator(evaluator=ms_marco_evaluator, config=config.evaluator)
    nlp.initialize()

    passages = [f"passage_{i}" for i in range(config.query_creator.size)]

    # for idx, m_pack in enumerate(nlp.process_dataset(
    # "./collection_and_queries/queries.dev.50.tsv")):
    for idx, m_pack in enumerate(nlp.process_dataset()):
        # import pdb
        # pdb.set_trace()
        # if (idx + 1) % 1000 == 0:
        #    print(f"Processed {idx+1} examples")
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

    # scores = nlp.evaluate()
    # print(scores)
