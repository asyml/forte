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
"""
The clinical ner processor
"""
import os
from typing import Dict, Any

from examples.Cliner.CliNER.code.predict import CliNERPredict
from forte.common import Resources
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.clinical import ClinicalEntityMention


class ClinicalNER(PackProcessor):
    def initialize(self, _: Resources, configs: Config):
        # Setup model path.
        # pylint: disable=attribute-defined-outside-init
        self.txt = configs.config_data
        self.output = configs.config_output
        self.model_path = configs.config_model
        self.format = 'i2b2'
        self.model = CliNERPredict(self.txt, self.output, self.model_path,
            self.format)

    def _process(self, input_pack: DataPack):
        with open(self.txt, 'r') as fin:
            doc = fin.readlines()

        self.model.predict()
        input_pack.pack_name = 'new_datapack_cliner'

        fname = os.path.splitext(os.path.basename(self.txt))[0] + '.' + 'con'
        with open(os.path.join(self.output, fname), "r", encoding="utf-8") \
                as con_file:
            con = con_file.readlines()
            ner_labels = []
            for line in con:
                labels = {}
                temp_labels = line[2:].strip().split('||')
                labels['type'] = temp_labels[1][3:-1]
                name_and_span = temp_labels[0].split('"')
                labels['name'] = name_and_span[1][0:]
                labels['span_begin'] = name_and_span[2].split()[0]
                labels['span_end'] = name_and_span[2].split()[1]
                labels['line_num'] = \
                    name_and_span[2].split()[0].split(':')[0]
                ner_labels.append(labels)

            offsets = []
            offset = 0
            text = ""
            text_lines = []

            for line in doc:
                text += line
                offsets.append(offset)  # the begin idx of each line
                offset += len(line)
                text_lines.append(line)

            for labels in ner_labels:
                line_num = int(labels['line_num']) - 1
                text_line = text_lines[line_num]
                new_text_line = text_line.lower()  # ignore the Uppercase and
                # lowercase
                word_begin = offsets[line_num] + new_text_line.index(
                    labels['name'])
                word_end = word_begin + len(labels['name'])

                entity = ClinicalEntityMention(input_pack, word_begin, word_end)
                entity.ner_type = labels['type']

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config['config_model'] = 'CliNER/models/train_full.model'
        config['config_output'] = 'CliNER/data/examples'
        config['config_data'] = 'CliNER/data/examples/ex_doc.txt'
        return config
