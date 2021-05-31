import sys
import time

import yaml
from mimic3_note_reader import Mimic3DischargeNoteReader

from examples.biobert_ner.bio_ner_predictor import BioBERTNERPredictor
from examples.biobert_ner.transformers_processor import BERTTokenizer
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.processors.writers import PackIdJsonPackWriter
from forte_wrapper.elastic import ElasticSearchPackIndexProcessor
from forte_wrapper.nltk import NLTKSentenceSegmenter


def main(input_path: str, output_path: str, max_packs: int = -1):
    pl = Pipeline[DataPack]()
    pl.set_reader(Mimic3DischargeNoteReader(),
                  config={'max_num_notes': max_packs})
    pl.add(NLTKSentenceSegmenter())

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
            'zip_pack': True
        }
    )

    pl.initialize()

    for idx, pack in enumerate(pl.process_dataset(input_path)):
        if (idx + 1) % 50 == 0:
            print(f"{time.strftime('%m-%d %H:%M')}: Processed {idx + 1} packs")


main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
