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
"""
The reader that reads SemEval Task8 data with annotations into Datapacks.
"""

import os
from typing import Any, Iterator, Dict
from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from forte.data.data_utils_io import dataset_path_iterator
from ft.onto.base_ontology import Sentence, RelationLink, EntityMention

__all__ = [
    "SemEvalTask8Reader"
]


class SemEvalTask8Reader(PackReader):
    r""":class:`SemEvalTask8Reader` is designed to read in
        SemEval Task-8 dataset. The data can be obtained
        here: http://www.kozareva.com/downloads.html


        `Hendrickx, Iris, et al. SemEval-2010 task 8: Multi-way
        classification of semantic relations between pairs of
        nominals.` https://www.aclweb.org/anthology/S10-1006.pdf

        An example of the dataset is
        '''
        8	"<e1>People</e1> have been moving back \
        into <e2>downtown</e2>."
        Entity-Destination(e1,e2)
        Comment:
        '''.

        This example will be converted to one `Sentence`,
        "People have been moving back into downtown."
        and one RelationLink,
        link = RelationLink(parent=People, child=downtown)
        link.rel_type = Entity-Destination
        into the DataPack.
    """

    def _cache_key_function(self, file_path: str) -> str:
        return os.path.basename(file_path)

    def _collect(self, *args, **kwargs) -> Iterator[Any]:
        # pylint: disable = unused-argument
        r"""args[0] should be the folder where
        the SemEval Task8 dataset is stored.
        Files ended with sem_eval_task8_file_extension (.txt)
        are exptected here.

        Args:
            args: args[0] is the directory to the dataset.

        Returns: Iterator over the file name (str).
        """
        sem_file_dir: str = args[0]
        return dataset_path_iterator(sem_file_dir,
                                     self.configs.sem_eval_task8_file_extension)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        pack: DataPack = DataPack()

        with open(file_path, 'r', encoding='utf8') as fp:
            txt = ""
            offset = 0

            while True:
                sent_line: str = fp.readline()
                if not sent_line:
                    break

                if len(sent_line.split()) == 0:
                    continue

                relation_line: str = fp.readline()
                # Command line is not used.
                _ = fp.readline()

                sent_line = sent_line[sent_line.find('"') + 1:
                                      sent_line.rfind('"')]
                index1 = sent_line.find("<e1>")
                index2 = sent_line.find("<e2>")
                # 5 is the length of "</e1>", include both <e1> and
                # </e1> when extracting the string.
                e1 = sent_line[index1:sent_line.find("</e1>") + 5]
                e2 = sent_line[index2:sent_line.find("</e2>") + 5]
                # Remove <e1> and </e1> in the sentence.
                sent_line = sent_line.replace(e1, e1[4:-5])
                sent_line = sent_line.replace(e2, e2[4:-5])
                # Remove <e1> and </e1> in e1.
                e1 = e1[4:-5]
                e2 = e2[4:-5]
                # Re-calculate the index after removing <e1>, </e1> in
                # in the sentence.
                if index1 < index2:
                    diff1 = 0
                    diff2 = 9
                else:
                    diff1 = 9
                    diff2 = 0
                index1 += offset - diff1
                index2 += offset - diff2

                Sentence(pack, offset, offset + len(sent_line))
                entry1 = EntityMention(pack, index1, index1 + len(e1))
                entry2 = EntityMention(pack, index2, index2 + len(e2))
                offset += len(sent_line) + 1
                txt += sent_line + " "

                pair = relation_line[relation_line.find("(") + 1:
                                     relation_line.find(")")]

                if "," in pair:
                    parent, _ = pair.split(",")
                    if parent == "e1":
                        relation = RelationLink(pack, entry1, entry2)
                    else:
                        relation = RelationLink(pack, entry2, entry1)
                    relation.rel_type = relation_line[:relation_line.find("(")]
                else:
                    # For "Other" relation, just set parent as e1
                    # set child as e2.
                    relation = RelationLink(pack, entry1, entry2)
                    relation.rel_type = relation_line.strip()

        pack.set_text(txt, replace_func=self.text_replace_operation)
        pack.pack_name = os.path.basename(file_path)

        yield pack

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        configs: Dict = super().default_configs()

        configs.update({
            'sem_eval_task8_file_extension': 'txt'
        })
        return configs
