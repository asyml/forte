import sys
import time

import yaml
from mimic3_note_reader import Mimic3DischargeNoteReader
from utterance_searcher import LastUtteranceSearcher

from forte.data.readers import RawDataDeserializeReader
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from forte.processors.writers import PackIdJsonPackWriter

from fortex.elastic import ElasticSearchPackIndexProcessor
from fortex.huggingface.bio_ner_predictor import BioBERTNERPredictor
from fortex.huggingface.transformers_processor import BERTTokenizer
from fortex.nltk import NLTKSentenceSegmenter

def main(input_path: str, output_path: str, max_packs: int = -1):
    pl = Pipeline[DataPack]()
    pl.set_reader(
        Mimic3DischargeNoteReader(), config={"max_num_notes": max_packs}
    )
    pl.add(NLTKSentenceSegmenter())

    config = yaml.safe_load(open("clinical_config.yml", "r"))
    config = Config(config, default_hparams=None)

    pl.add(BERTTokenizer(), config=config.BERTTokenizer)
    pl.add(BioBERTNERPredictor(), config=config.BioBERTNERPredictor)
    pl.add(ElasticSearchPackIndexProcessor())

    pl.initialize()
    for idx, _ in enumerate(pl.process_dataset(input_path)):
        if (idx + 1) % 50 == 0:
            print(f"{time.strftime('%m-%d %H:%M')}: Processed {idx + 1} packs")
    
    remote_pl = Pipeline[DataPack]()
    remote_pl.set_reader(RawDataDeserializeReader())
    remote_pl.add(LastUtteranceSearcher(), config=config.LastUtteranceSearcher)
    remote_pl.serve(port=config.Remote.port, input_format=config.Remote.input_format, service_name=config.Remote.service_name)

main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
