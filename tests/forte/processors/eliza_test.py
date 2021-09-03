"""This module tests the Eliza processor."""
import unittest

from ddt import ddt, data

from forte.data.common_entry_utils import create_utterance, get_last_utterance
from forte.data.data_pack import DataPack
from forte.data.readers import StringReader
from forte.pipeline import Pipeline
from forte.processors.base import PackProcessor
from forte.processors.nlp import ElizaProcessor
from ft.onto.base_ontology import Utterance


class UserSimulator(PackProcessor):
    """
    A simulated processor that will generate utterance based on the config.
    """

    def _process(self, input_pack: DataPack):
        create_utterance(input_pack, self.configs.user_input, "user")

    @classmethod
    def default_configs(cls):
        return {"user_input": ""}


@ddt
class TestElizaProcessor(unittest.TestCase):
    def setUp(self):
        self.nlp = Pipeline[DataPack]()
        self.nlp.set_reader(StringReader())

    @data(
        [
            "I would like to have a chat bot.",
            "You say you would like to have a chat bot ?",
        ],
        ["bye", "Goodbye.  Thank you for talking to me."],
    )
    def test_eliza_processor(self, input_output_pair):
        i_str, o_str = input_output_pair
        self.nlp.add(UserSimulator(), config={"user_input": i_str})
        self.nlp.add(ElizaProcessor())
        self.nlp.initialize()
        res: DataPack = self.nlp.process("")

        u = get_last_utterance(res, "ai")

        self.assertEqual(len([_ for _ in res.get(Utterance)]), 2)

        self.assertEqual(u.text, o_str)


if __name__ == "__main__":
    unittest.main()
