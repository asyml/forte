import yaml

from termcolor import colored
import texar.torch as tx
from forte.data.readers import MultiPackTerminalReader
from forte.processors import ElasticSearchQueryCreator, ElasticSearchProcessor
from forte.pipeline import Pipeline

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yml", "r"))
    config = tx.HParams(config, default_hparams=None)
    nlp = Pipeline()
    nlp.set_reader(reader=MultiPackTerminalReader(), config=config.reader)
    nlp.add_processor(processor=ElasticSearchQueryCreator(),
                      config=config.query_creator)
    nlp.add_processor(processor=ElasticSearchProcessor(), config=config.indexer)
    nlp.initialize()

    for m_pack in nlp.process_dataset():
        for name, pack in zip(m_pack.pack_names, m_pack.packs):
            print(colored(name, "green"), ":", pack.text)
