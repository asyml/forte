"""
Unit tests for remote processor.
"""
from forte.data.readers import RawDataDeserializeReader
from forte.processors.misc import RemoteProcessor
from forte.data.common_entry_utils import get_last_utterance
from forte.data.data_pack import DataPack
from forte.data.readers import StringReader
from forte.pipeline import Pipeline
from forte.processors.nlp import ElizaProcessor
from ft.onto.base_ontology import Utterance
from advanced_pipeline_test import (
    DummyProcessor,
    TEST_RECORDS_1,
    TEST_RECORDS_2,
    UserSimulator,
)
from ddt import ddt, data
import unittest


@ddt
class AdvancedPipelineTest(unittest.TestCase):
    """
    Test RemoteProcessor. Here we use eliza
    pipeline as an example, and all the testcases below are refactored from
    `eliza_test.py`.
    """

    @data(
        [
            "I would like to have a chat bot.",
            "You say you would like to have a chat bot ?",
        ],
        ["bye", "Goodbye.  Thank you for talking to me."],
    )
    def test_remote_processor(self, input_output_pair):
        """
        Verify RemoteProcessor.
        """
        i_str, o_str = input_output_pair
        service_name: str = "test_service_name"
        input_format: str = "DataPack"

        # Build service pipeline
        serve_pl: Pipeline[DataPack] = Pipeline[DataPack]()
        serve_pl.set_reader(RawDataDeserializeReader())
        serve_pl.add(DummyProcessor(expected_records=TEST_RECORDS_1))
        serve_pl.add(UserSimulator(), config={"user_input": i_str})
        serve_pl.add(DummyProcessor(output_records=TEST_RECORDS_2))
        serve_pl.add(ElizaProcessor())
        serve_pl.initialize()

        # Configure RemoteProcessor into test mode
        remote_processor: RemoteProcessor = RemoteProcessor()
        remote_processor.set_test_mode(
            serve_pl._remote_service_app(
                service_name=service_name, input_format=input_format
            )
        )

        # Build test pipeline
        test_pl: Pipeline[DataPack] = Pipeline[DataPack](
            do_init_type_check=True
        )
        test_pl.set_reader(StringReader())
        test_pl.add(DummyProcessor(output_records=TEST_RECORDS_1))
        test_pl.add(
            remote_processor,
            config={
                "validation": {
                    "do_init_type_check": True,
                    "input_format": input_format,
                    "expected_name": service_name,
                }
            },
        )
        test_pl.add(DummyProcessor(expected_records=TEST_RECORDS_2))
        test_pl.initialize()

        # Verify output
        res: DataPack = test_pl.process("")
        utterance = get_last_utterance(res, "ai")
        self.assertEqual(len([_ for _ in res.get(Utterance)]), 2)
        self.assertEqual(utterance.text, o_str)
