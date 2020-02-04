# Copyright 2020 The Forte Authors. All Rights Reserved.
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

import subprocess
import sys
from importlib import import_module
from distutils.util import strtobool
import logging

from scispacy.umls_linking import UmlsEntityLinker
from spacy.language import Language
from texar.torch import HParams

from forte.common import ProcessorConfigError
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import LinkedMention

logger = logging.getLogger(__name__)

__all__ = [
    "ScispaCyUMLSEntityLinker"
]

BASE_URL_STRING = 'https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/'

# install('https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz')


class ScispaCyUMLSEntityLinker(PackProcessor):
    """
    ScispaCy Processor that links entities from the UMLS KB.
    """
    def __init__(self):
        super().__init__()
        self.model_name: str = ""
        self.model_version: str = ""
        self.nlp: Language = None
        self.resolve_abbr: bool = False
        self.overwrite_entries: bool = False
        self.allow_parallel_entries: bool = True

    def create_url(self):
        return BASE_URL_STRING + f'v/{self.model_version}/{self.model_name}' \
            f'-{self.model_version}.tar.gz'

    @staticmethod
    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    def set_up(self):
        self.install(self.create_url())
        scispacy_model = import_module(self.model_name)
        self.nlp = scispacy_model.load()
        linker = UmlsEntityLinker(resolve_abbreviations=self.resolve_abbr)
        self.nlp.add_pipe(linker)

    # pylint: disable=unused-argument
    def initialize(self, resource: Resources,
                   configs: HParams):
        self.model_name = configs.model_name or \
                          self.default_configs()['model_name']
        self.model_version = configs.model_version or self.default_configs()['model_version']
        self.overwrite_entries = configs.overwrite_entries
        self.allow_parallel_entries = configs.allow_parallel_entries
        if self.overwrite_entries:
            logger.warning("`overwrite_entries` is set to True, this means "
                           "that the entries of the same type as produced by "
                           "this processor will be overwritten if found.")
            if self.allow_parallel_entries:
                logger.warning('Both `overwrite_entries` (whether to overwrite '
                               'the entries of the same type as produced by '
                               'this processor) and '
                               '`allow_parallel_entries` (whether to allow '
                               'similar new entries when they already exist) '
                               'are True, all existing conflicting entries '
                               'will be deleted.')
        else:
            if not self.allow_parallel_entries:
                logger.warning('Both `overwrite_entries` (whether to overwrite '
                               'the entries of the same type as produced by '
                               'this processor) and '
                               '`allow_parallel_entries` (whether to allow '
                               'similar new entries when they already exist) '
                               'are False, processor will only run if there '
                               'are no existing conflicting entries.')

        try:
            self.resolve_abbr = strtobool(configs.resolve_abbreviations)
        except ValueError:
            self.resolve_abbr = False
        self.set_up()

    @staticmethod
    def default_configs():
        """
        This defines a basic Hparams structure for spaCy.
        Returns:

        """
        return {
            'model_name': 'en_core_sci_sm',
            'model_version': '0.2.4',
            'resolve_abbreviations': False,
            'overwrite_entries': False,
            'allow_parallel_entries': True
        }

    def _process(self, input_pack: DataPack):
        # handle existing entries
        self._process_existing_entries(input_pack)
        doc = input_pack.text

        processed_doc = self.nlp(doc)
        for entity in processed_doc.ents:
            linked_entity = LinkedMention(input_pack, entity.start_char,
                                          entity.end_char)
            linked_entity.kb('UMLS')
            cand_entities = {}
            for umls_ent in entity._.umls_ents:
                cand_entities[umls_ent[0]] = umls_ent[1]  # K = CUI, V = score

            linked_entity.linked_kb_ids(cand_entities)
            input_pack.add_entry(linked_entity)

    def _process_existing_entries(self, input_pack):
        lm_exist = any(True for _ in input_pack.get(LinkedMention))

        if lm_exist:
            if not self.overwrite_entries:
                if not self.allow_parallel_entries:
                    raise ProcessorConfigError("Found existing entries, either "
                                               "`overwrite_entries` or "
                                               "`allow_parallel_entries` "
                                               "should be True")
            else:
                # delete existing tokens and dependencies
                for entry in input_pack.get(LinkedMention):
                        input_pack.delete_entry(entry)







