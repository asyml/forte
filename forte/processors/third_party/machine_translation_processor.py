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
import uuid
from os import getenv
from typing import Dict, Any
from urllib.parse import urlencode
import os

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.multi_pack import MultiPack
from forte.processors.base import MultiPackProcessor, PackProcessor
from forte.utils import create_import_error_msg
from ft.onto.base_ontology import Document, Utterance

try:
    from transformers import (  # pylint:disable=import-outside-toplevel
        T5Tokenizer,
        T5ForConditionalGeneration,
    )
except ImportError as err:
    raise ImportError(
        create_import_error_msg(
            "transformers", "data_aug", "Machine Translator"
        )
    ) from err

__all__ = ["MicrosoftBingTranslator", "MachineTranslationProcessor"]


class MicrosoftBingTranslator(MultiPackProcessor):
    r"""This processor translates text from one language to another using
    Microsoft Bing Translate APIs. To use this processor, 'MICROSOFT_API_KEY'
    should be set as an environment variable to Microsoft Translator API
    subscription key.

    """

    def __init__(self) -> None:
        super().__init__()
        self.microsoft_translate_url = (
            "https://api.cognitive.microsofttranslator.com/translate"
        )
        self.microsoft_headers_content_type = "application/json"
        self.microsoft_headers = {
            "Ocp-Apim-Subscription-Key": getenv("MICROSOFT_API_KEY"),
            "Content-type": self.microsoft_headers_content_type,
            "X-ClientTraceId": str(uuid.uuid4()),
        }

    # pylint: disable=unused-argument
    def initialize(self, resources: Resources, configs: Config):
        r"""Initialize the processor with `resources` and `configs`. This method
        is called by the pipeline during the initialization.

        Args:
            resources (Resources): An object of class
                :class:`forte.common.Resources` that holds references to objects
                that can be shared throughout the pipeline.
            configs (Config): A configuration to initialize the
                processor. This processor is expected to hold the following
                (key, value) pairs

                - `"src_language"` (str): Source language for the translation
                - `"target_language"` (str): Target language for the translation
                - `"in_pack_name"` (str): Pack name to be used to fetch input
                  datapack.
                - `"out_pack_name"` (str): Pack name to be used to fetch output
                  datapack

        """
        self.resources = resources

        if configs:
            self.src_language = configs.src_language
            self.target_language = configs.target_language
            self.in_pack_name = configs.in_pack_name
            self.out_pack_name = configs.out_pack_name

    def _process(self, input_pack: MultiPack):
        try:
            import requests  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError(
                create_import_error_msg(
                    "requests", "data_aug", "data augment support"
                )
            ) from e
        query = input_pack.get_pack(self.in_pack_name).text
        params = "?" + urlencode(
            {
                "api-version": "3.0",
                "from": self.src_language,
                "to": [self.target_language],
            },
            doseq=True,
        )
        microsoft_constructed_url = self.microsoft_translate_url + params

        response = requests.post(
            microsoft_constructed_url,
            headers=self.microsoft_headers,
            json=[{"text": query}],
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(response.json()["error"]["message"])

        text = response.json()[0]["translations"][0]["text"]
        pack: DataPack = input_pack.add_pack(self.out_pack_name)
        pack.set_text(text=text)

        Document(pack, 0, len(text))
        Utterance(pack, 0, len(text))

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "src_language": "en",
            "target_language": "de",
            "in_pack_name": "doc_0",
            "out_pack_name": "response",
        }


class MachineTranslationProcessor(PackProcessor):
    """
    Translate the input text and output to a file.
    """

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        # Initialize the tokenizer and model
        model_name: str = self.configs.pretrained_model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.task_prefix = "translate English to German: "
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if not os.path.isdir(self.configs.output_folder):
            os.mkdir(self.configs.output_folder)

    def _process(self, input_pack: DataPack):
        file_name: str = os.path.join(
            self.configs.output_folder, os.path.basename(input_pack.pack_name)
        )

        # en2de machine translation
        inputs = self.tokenizer(
            [
                self.task_prefix + sentence
                for sentence in input_pack.text.split("\n")
            ],
            return_tensors="pt",
            padding=True,
        )

        output_sequences = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,
        )

        outputs = self.tokenizer.batch_decode(
            output_sequences, skip_special_tokens=True
        )

        # Write output to the specified file
        with open(file=file_name, mode="w", encoding="utf-8") as f:
            f.write("\n".join(outputs))

    @classmethod
    def default_configs(cls) -> Dict:
        return {
            "pretrained_model": "t5-small",
            "output_folder": "mt_test_output",
        }
