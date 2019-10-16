# pylint: disable=attribute-defined-outside-init
"""
The reader that reads text data from into Multipack.
"""
import logging
from typing import Iterator

from termcolor import colored

from texar.torch import HParams
from forte.common.resources import Resources
from forte import config
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.data.ontology import base_ontology
from forte.data.readers.base_reader import MultiPackReader

logger = logging.getLogger(__name__)

__all__ = [
    "MultiPackTerminalReader"
]


class MultiPackTerminalReader(MultiPackReader):
    r"""
    A reader designed to read text from the terminal
    """

    def __init__(self):
        super().__init__()
        self._ontology = base_ontology
        self.define_output_info()

    # pylint: disable=unused-argument
    def initialize(self, resource: Resources, configs: HParams):
        self.resource = resource

    def define_output_info(self):
        return {
            self._ontology.Document: [],
        }

    # pylint: disable=no-self-use,unused-argument
    def _cache_key_function(self, collection) -> str:
        return "cached_string_file"

    # pylint: disable=no-self-use
    def _collect(self) -> Iterator[str]:  # type: ignore
        """
        data_strings should be of type `List[str]`
        which is the list of raw text strings to iterate over
        """
        # This allows the user to pass in either one single string or a list of
        # strings.
        while True:
            try:
                data = input(colored("Enter your query here: ", 'green'))
                if len(data) == 0:
                    continue
                yield data
            except EOFError:
                print()
                break

    def parse_pack(self, data_source: str) -> MultiPack:
        """
        Takes a raw string and converts into a MultiPack

        Args:
            data_source: str that contains text of a document

        Returns: MultiPack containing a datapack for the current query
        """
        config.working_component = self.component_name

        multi_pack = MultiPack()

        # use context to build the query
        if self.resource.get("user_utterance"):
            user_pack = self.resource.get("user_utterance")[-1]
            multi_pack.update_pack({"user_utterance": user_pack})

        if self.resource.get("bot_utterance"):
            bot_pack = self.resource.get("bot_utterance")[-1]
            multi_pack.update_pack({"bot_utterance": bot_pack})

        pack = DataPack()
        utterance = base_ontology.Utterance(pack, 0, len(data_source))
        pack.add_or_get_entry(utterance)

        pack.set_text(data_source, replace_func=self.text_replace_operation)
        multi_pack.update_pack({"query": pack})
        config.working_component = None

        return multi_pack
