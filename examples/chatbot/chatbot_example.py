import yaml

from termcolor import colored
import torch
from texar.torch import HParams

from forte.data.readers import MultiPackTerminalReader
from forte.common.resources import Resources
from forte.pipeline import Pipeline
from forte.processors import SRLPredictor
from forte.processors.machine_translation_processor import \
    MachineTranslationProcessor
from forte.processors.query_creator import QueryCreator
from forte.processors.search_processor import SearchProcessor
from forte.data.selector import NameMatchSelector
from forte.processors.nltk_processors import \
    (NLTKSentenceSegmenter, NLTKWordTokenizer, NLTKPOSTagger)

from ft.onto.base_ontology import PredicateLink, Sentence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    config = yaml.safe_load(open("config.yml", "r"))
    config = HParams(config, default_hparams=None)

    resource = Resources()
    query_pipeline = Pipeline(resource=resource)
    query_pipeline.set_reader(
        reader=MultiPackTerminalReader(), config=config.reader)

    query_pipeline.add_processor(
        processor=MachineTranslationProcessor(), config=config.translator)
    query_pipeline.add_processor(
        processor=QueryCreator(), config=config.query_creator)
    query_pipeline.add_processor(
        processor=SearchProcessor(), config=config.indexer)
    query_pipeline.add_processor(
        processor=NLTKSentenceSegmenter(),
        selector=NameMatchSelector(
            select_name=config.indexer.response_pack_name[0]))
    query_pipeline.add_processor(
        processor=NLTKWordTokenizer(),
        selector=NameMatchSelector(
            select_name=config.indexer.response_pack_name[0]))
    query_pipeline.add_processor(
        processor=NLTKPOSTagger(),
        selector=NameMatchSelector(
            select_name=config.indexer.response_pack_name[0]))
    query_pipeline.add_processor(
        processor=SRLPredictor(), config=config.SRL,
        selector=NameMatchSelector(
            select_name=config.indexer.response_pack_name[0]))
    query_pipeline.add_processor(
        processor=MachineTranslationProcessor(), config=config.back_translator)

    query_pipeline.initialize()

    for m_pack in query_pipeline.process_dataset():

        # update resource to be used in the next conversation
        query_pack = m_pack.get_pack(config.translator.in_pack_name)
        if resource.get("user_utterance"):
            resource.get("user_utterance").append(query_pack)
        else:
            resource.update(user_utterance=[query_pack])

        response_pack = m_pack.get_pack(config.back_translator.in_pack_name)

        if resource.get("bot_utterance"):
            resource.get("bot_utterance").append(response_pack)
        else:
            resource.update(bot_utterance=[response_pack])

        english_pack = m_pack.get_pack("pack")
        print(colored("English Translation of the query: ", "green"),
              english_pack.text, "\n")
        pack = m_pack.get_pack(config.indexer.response_pack_name[0])
        print(colored("Retrieved Document", "green"), pack.text, "\n")
        print(colored("German Translation", "green"),
              m_pack.get_pack("response").text, "\n")
        for sentence in pack.get(Sentence):
            sent_text = sentence.text
            print(colored("Sentence:", 'red'), sent_text, "\n")

            print(colored("Semantic role labels:", 'red'))
            for link in pack.get(PredicateLink, sentence):
                parent = link.get_parent()
                child = link.get_child()
                print(f"  - \"{child.text}\" is role {link.arg_type} of "
                      f"predicate \"{parent.text}\"")
            print()

            input(colored("Press ENTER to continue...\n", 'green'))


if __name__ == "__main__":
    main()
