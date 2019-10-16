# pylint: disable=attribute-defined-outside-init
from urllib.parse import urlencode
from os import getenv
import uuid
from typing import Optional
import requests

from texar.torch.hyperparams import HParams

from forte.common.resources import Resources
from forte.data import DataPack, MultiPack
from forte.data.ontology import base_ontology
from forte.processors.base import MultiPackProcessor, ProcessInfo

__all__ = [
    "MachineTranslationProcessor"
]


class MachineTranslationProcessor(MultiPackProcessor):
    r"""This processor is used to translate text from one language to another
    """

    def __init__(self) -> None:
        super().__init__()
        self._ontology = base_ontology
        self.microsoft_translate_url = \
            "https://api.cognitive.microsofttranslator.com/translate"
        self.microsoft_headers_content_type = 'application/json'
        self.microsoft_headers = {
            'Ocp-Apim-Subscription-Key': getenv('MICROSOFT_API_KEY'),
            'Content-type': self.microsoft_headers_content_type,
            'X-ClientTraceId': str(uuid.uuid4())
        }

    # pylint: disable=unused-argument
    def initialize(self, resources: Optional[Resources],
                   configs: Optional[HParams]):
        if configs:
            self.src_language = configs.src_language
            self.target_language = configs.target_language
            self.in_pack_name = configs.in_pack_name
            self.out_pack_name = configs.out_pack_name

    # pylint: disable=no-self-use
    def _define_input_info(self) -> ProcessInfo:
        input_info: ProcessInfo = {

        }

        return input_info

    # pylint: disable=no-self-use
    def _define_output_info(self) -> ProcessInfo:
        output_info: ProcessInfo = {

        }

        return output_info

    def _process(self, input_pack: MultiPack):
        query = input_pack.get_pack(self.in_pack_name).text
        params = '?' + urlencode(
            {'api-version': '3.0',
             'from': self.src_language,
             'to': [self.target_language]}, doseq=True)
        microsoft_constructed_url = self.microsoft_translate_url + params

        response = requests.post(
            microsoft_constructed_url, headers=self.microsoft_headers,
            json=[{"text": query}])

        if response.status_code != 200:
            raise RuntimeError(response.json()['error']['message'])

        text = response.json()[0]["translations"][0]["text"]
        pack = DataPack()

        document = base_ontology.Document(pack, 0, len(text))
        utterance = base_ontology.Utterance(pack, 0, len(text))
        pack.add_or_get_entry(document)
        pack.add_or_get_entry(utterance)

        pack.set_text(text=text)
        input_pack.update_pack({self.out_pack_name: pack})
