# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=attribute-defined-outside-init
from urllib.parse import urlencode
from os import getenv
import uuid
from typing import Optional
import requests

from texar.torch.hyperparams import HParams

from forte.common.resources import Resources
from forte.data import DataPack, MultiPack
from forte.processors.base import MultiPackProcessor

from ft.onto.base_ontology import Document, Utterance

__all__ = [
    "MachineTranslationProcessor"
]


class MachineTranslationProcessor(MultiPackProcessor):
    r"""This processor is used to translate text from one language to another
    """

    def __init__(self) -> None:
        super().__init__()
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

        document = Document(pack, 0, len(text))
        utterance = Utterance(pack, 0, len(text))
        pack.add_or_get_entry(document)
        pack.add_or_get_entry(utterance)

        pack.set_text(text=text)
        input_pack.update_pack({self.out_pack_name: pack})
