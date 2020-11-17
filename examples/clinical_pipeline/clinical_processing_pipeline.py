import sys

import yaml
from mimic3_note_reader import Mimic3DischargeNoteReader

from examples.biobert_ner.bio_ner_predictor import BioBERTNERPredictor
from examples.biobert_ner.transformers_processor import BERTTokenizer
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.processors.ir import ElasticSearchPackIndexProcessor
from forte.processors.nltk_processors import NLTKSentenceSegmenter, \
    NLTKWordTokenizer, NLTKPOSTagger, NLTKNER
from forte.processors.writers import PackIdJsonPackWriter


def main(input_path: str, output_path: str, max_packs: int = -1):
    pl = Pipeline[DataPack]()
    pl.set_reader(Mimic3DischargeNoteReader())
    pl.add(NLTKSentenceSegmenter())
    pl.add(NLTKWordTokenizer())
    pl.add(NLTKPOSTagger())
    pl.add(NLTKNER())

    config = yaml.safe_load(open("bio_ner_config.yml", "r"))
    config = Config(config, default_hparams=None)

    pl.add(BERTTokenizer(), config=config.BERTTokenizer)
    pl.add(BioBERTNERPredictor(), config=config.BioBERTNERPredictor)
    pl.add(ElasticSearchPackIndexProcessor())

    pl.add(
        PackIdJsonPackWriter(),
        {
            'output_dir': output_path,
            'indent': 2,
            'overwrite': True,
            'drop_record': True,
        }
    )

    pl.initialize()

    print()
    for idx, pack in enumerate(pl.process_dataset(input_path)):
        if (idx + 1) % 10 == 0:
            print(f"\rProcessed {idx + 1} packs")
        if 0 < max_packs < idx:
            break

    print()


main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
