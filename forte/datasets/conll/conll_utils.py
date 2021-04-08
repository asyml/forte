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
Here we define some utility functions for CoNLL evaluation datasets change.
We can add other datasets conversion function for CoNLL here in the future.
"""
from typing import List, Dict, Optional, Type

from forte.data.data_pack import DataPack
from forte.data.base_pack import PackType
from forte.data.ontology import Annotation
from forte.data.extractors.utils import bio_tagging
from ft.onto.base_ontology import Sentence


def post_edit(element: List[Optional[str]]) -> str:
    if element[0] is None:
        return "O"
    return "%s-%s" % (element[1], element[0])


def get_tag(pack: DataPack,
            sentence: Annotation,
            tagging_unit: Type[Annotation],
            entry: Type[Annotation],
            attribute: str) -> List[str]:
    tag = bio_tagging(pack,
                        sentence,
                        tagging_unit,
                        entry,
                        attribute)
    tag = [post_edit(x) for x in tag]
    return tag


def write_tokens_to_file(pred_pack: PackType,
                         refer_pack: PackType,
                         refer_request: Dict,
                         tagging_unit: Type[Annotation],
                         entry: Type[Annotation],
                         attribute: str,
                         output_file: str):
    opened_file = open(output_file, "a+")
    for refer_data, pred_sent, refer_sent in zip(
        refer_pack.get_data(**refer_request),
        pred_pack.get(Sentence),
        refer_pack.get(Sentence)):

        refer_tag = get_tag(refer_pack,
                            refer_sent,
                            tagging_unit,
                            entry,
                            attribute)
        pred_tag = get_tag(pred_pack,
                           pred_sent,
                           tagging_unit,
                           entry,
                           attribute)

        words = refer_data["Token"]["text"]

        for i, (word, tgt, pred) in \
                enumerate(zip(words, refer_tag, pred_tag), 1):
            opened_file.write(
                "%d %s %s %s\n" % (i, word, tgt, pred)
            )
        opened_file.write("\n")
    opened_file.close()
