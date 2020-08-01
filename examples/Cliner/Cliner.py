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
import codecs
import os
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.clinical import ClinicalEntityMention

from examples.Cliner.CliNER.code.predict import CliNERPredict


class ClinicalNER(PackProcessor):
    def initialize(self):
        # Setup model path.
        # pylint: disable=attribute-defined-outside-init
        self.txt = os.path.join('CliNER/data/examples/ex_doc.txt')
        self.output = os.path.join('CliNER/data/test_predictions')
        self.model_path = os.path.join('CliNER/models/train_full.model')
        self.format = 'i2b2'
        self.model = CliNERPredict(self.txt, self.output, self.model_path,
                                   self.format)

    def _process(self, input_pack: DataPack):
        with open(self.txt, 'r') as fin:
            doc = fin.readlines()

        self.model.predict()

        con = codecs.open(os.path.join(self.output, 'ex_doc.con'),
                          "r",
                          encoding="utf8")
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
            offsets.append(offset)  # the begin of the text
            offset += len(line) + 1
            text_lines.append(line)

        for labels in ner_labels:
            line_num = int(labels['line_num']) - 1
            text_line = text_lines[line_num]
            span_begin = \
                text_line.split()[int(labels['span_begin'].split(':')[1])]
            word_begin = offsets[line_num] + text_line.index(span_begin)
            word_end = word_begin + len(labels['name'])
            entity = ClinicalEntityMention(input_pack, word_begin, word_end)
            entity.cliner_type = labels['type']
