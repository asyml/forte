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
The reader that reads prodigy text data with annotations into Datapacks.
"""

import os
import json
from typing import Any, Iterator
from forte.common.exception import PackDataException
from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader
from forte.data.data_utils_io import dataset_path_iterator
from ft.onto.base_ontology import Sentence, RelationLink, EntityMention

__all__ = [
    "SemEvalTask8Reader"
]


class SemEvalTask8Reader(PackReader):
    r""":class:`SemEvalTask8Reader` is designed to read in Sem Eval Dataset for task8.
    
        An example of the dataset
        '''
        8	"<e1>People</e1> have been moving back into <e2>downtown</e2>."
        Entity-Destination(e1,e2)
        Comment:
        '''

    """
    def _cache_key_function(self, file_path: str) -> str:
        return os.path.basename(file_path)

    def _collect(self,  # type: ignore
                 sem_file_dir: str) -> Iterator[Any]:
        return dataset_path_iterator(sem_file_dir,  
            self.configs.sem_eval_task8_file_extension)

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        pack = self.new_pack()
        fp = open(file_path, 'r', encoding='utf8')
        txt = ""
        offset = 0

        while True:
            sent_line = fp.readline()
            if not sent_line:
                break
            
            if len(sent_line.split()) != 0:
                relation_line = fp.readline()
                comment_line = fp.readline()

                if not relation_line or not relation_line:
                    raise PackDataException("""The Sem_eval data is incorrect with \n
                                                sent_line: %s\n
                                                relation_line: %s\n
                                                comment_line: %s\n""" % (
                                                    sent_line, relation_line, comment_line
                                                ))

                # try:
                sent_line = sent_line[sent_line.find('"')+1:sent_line.rfind('"')]
                
                e1 = sent_line[sent_line.find("<e1>")+4:sent_line.find("</e1>")]
                e2 = sent_line[sent_line.find("<e2>")+4:sent_line.find("</e2>")]

                sent_line = sent_line.replace(e1, e1[4:-5])
                sent_line = sent_line.replace(e2, e2[4:-5])
                e1 = e1[4:-5]
                e2 = e2[4:-5]

                Sentence(pack, offset, offset + len(sent_line))
                offset += len(sent_line) + 1
                txt += sent_line + " "

                entry1 = EntityMention(pack, offset, offset + len(e1))
                offset += len(e1) + 1
                txt += e1 + " "

                entry2 = EntityMention(pack, offset, offset + len(e2))
                offset += len(e2) + 1
                txt += e2 + " "           

                pair = relation_line[relation_line.find("(")+1:relation_line.find(")")]

                if "," in pair:
                    parent, _ = pair.split(",")
                    
                    # TODO: How to add relation? 
                    # relation = relation_line[:relation_line.find("(")]
                    if parent == "e1":
                        RelationLink(pack, entry1, entry2)
                    else:
                        RelationLink(pack, entry2, entry1)
                    offset += len(relation_line) + 1
                    txt += relation_line + " "
                else:
                    #TODO
                # except:
                #     print("error!")
                #     print("sent", sent_line)
                #     print("relation", relation_line)
        
        pack.set_text(txt, replace_func=self.text_replace_operation)
        pack.pack_name = file_path

        yield pack

    @classmethod
    def default_configs(cls):
        configs = super().default_configs()

        configs.update({
            'sem_eval_task8_file_extension': 'txt'
        })
        return configs