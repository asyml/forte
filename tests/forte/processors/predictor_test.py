#  Copyright 2020 The Forte Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import unittest
from dataclasses import dataclass
from typing import Optional

from forte.data.data_pack import DataPack
from forte.data.ontology import Generics
from forte.data.readers.conll03_reader import CoNLL03Reader
from forte.evaluation.ner_evaluator import CoNLLNEREvaluator
from forte.pipeline import Pipeline
from forte.processors.base.batch_processor import Predictor
from ft.onto.base_ontology import Sentence, EntityMention

FAKEOUTPUT = 2


@dataclass
class NewType(Generics):
    """A dummy generic type to check the correctness of pipeline execution."""

    value: Optional[str] = None

    def __init__(self, pack, value):
        super().__init__(pack)
        self.value = value


class DummyModel:
    def __call__(self, batch):
        text_feature = batch["text_tag"]["data"]
        return {
            "ner_tag": [
                [FAKEOUTPUT for j in range(len(text_feature[0]))]
                for i in range(len(text_feature))
            ]
        }


class NERPredictor(Predictor):
    def predict(self, batch):
        return self.model(batch)


class PredictorTest(unittest.TestCase):
    def setUp(self):
        # Setup path
        self.dataset_path: str = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.sep.join([".."] * 3),
                "data_samples/conll03",
            )
        )

    def test_Predictor(self):
        pipeline = Pipeline[DataPack]()
        pipeline.set_reader(CoNLL03Reader())
        pipeline.initialize()

        text_extractor_name = (
            "forte.data.extractors.attribute_extractor.AttributeExtractor"
        )
        text_extractor_config = {
            "need_pad": True,
            "entry_type": "ft.onto.base_ontology.Token",
            "attribute": "text",
        }

        ner_extractor_name = (
            "forte.data.extractors.seqtagging_extractor.BioSeqTaggingExtractor"
        )
        ner_extractor_config = {
            "entry_type": "ft.onto.base_ontology.EntityMention",
            "attribute": "ner_type",
            "tagging_unit": "ft.onto.base_ontology.Token",
            "need_pad": True,
            "context_type": "ft.onto.base_ontology.Sentence",
        }

        model = DummyModel()

        predictor_pipeline = Pipeline[DataPack]()
        predictor_pipeline.set_reader(CoNLL03Reader())

        predictor_config = {
            "context_type": "ft.onto.base_ontology.Sentence",
            "feature_scheme": {
                "text_tag": {
                    "type": "data_input",
                    "extractor": {
                        "class_name": text_extractor_name,
                        "config": text_extractor_config,
                    },
                },
                "ner_tag": {
                    "type": "data_output",
                    "extractor": {
                        "class_name": ner_extractor_name,
                        "config": ner_extractor_config,
                    },
                },
            },
            "do_eval": True,
            "batcher": {
                "batch_size": 2,
            },
        }
        # dummy = DummyRelationExtractor()
        # config = {"batcher": {"batch_size": 5}}
        evaluator_config = {
            "entry_type": "ft.onto.base_ontology.EntityMention",
            "attribute": "ner_type",
            "tagging_unit": "ft.onto.base_ontology.Token",
        }
        predictor = NERPredictor()
        predictor.load(model)
        predictor_pipeline.add(predictor, predictor_config)
        predictor_pipeline.add(CoNLLNEREvaluator(), evaluator_config)
        predictor_pipeline.initialize()

        text_extractor = predictor._request["schemes"]["text_tag"]["extractor"]
        ner_extractor = predictor._request["schemes"]["ner_tag"]["extractor"]

        for pack in pipeline.process_dataset(self.dataset_path):
            text_extractor.update_vocab(pack)

        for pack in pipeline.process_dataset(self.dataset_path):
            ner_extractor.update_vocab(pack)

        expected_ners = [
            ner_extractor.id2element(FAKEOUTPUT)[0] for _ in range(30)
        ]

        for pack in predictor_pipeline.process_dataset(self.dataset_path):
            for instance in pack.get(Sentence):
                ners = [
                    e.ner_type for e in list(pack.get(EntityMention, instance))
                ]
                self.assertListEqual(ners, expected_ners)


if __name__ == "__main__":
    unittest.main()
