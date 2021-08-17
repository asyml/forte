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
from typing import List, Dict, Optional, Type, Tuple

from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
from forte.data.extractors.utils import bio_tagging
from ft.onto.base_ontology import Sentence


def post_edit(element: Tuple[Optional[str], str]) -> str:
    r"""Post process the tags. Convert it from tuple
    to string.

    Args:
        element Tuple[Optional[str], str]:
            a tuple of BIO tag with element[0]
            as tag label and element[1] as
            BIO, e.g. `("PER", "B")`.

    Returns:
        BIO tag in string format, e.g. `"B-PER"`.

    """
    if element[0] is None:
        return "O"
    return "%s-%s" % (element[1], element[0])


def get_tag(
    pack: DataPack,
    instance: Annotation,
    tagging_unit: Type[Annotation],
    entry_type: Type[Annotation],
    attribute: str,
) -> List[str]:
    r"""Align entries to tagging units, and convert it to string format.

    Args:
        pack (DataPack): The datapack that contains the current
            instance.

        instance (Annotation): The instance from which the
            extractor will extractor feature. For example, an instance of
            Sentence type, which mean the tagging sequence comes from
            one sentence.

        tagging_unit (Type[Annotation]): The type of tagging unit that entry
            tag should align to. For example, it can be Token, which means
            returned tags should aligned to tokens in one sentence.

        entry_type (Type[Annotation]): The type of entry that contains tags. For
            example, it can be EntityMethion, which means tags comes from the
            EntityMention of one sentence. Note that the number of EntityMention
            can be different from the number of Token. That is why we need to
            use BIO tagging to aglin them.

        attribute (str): A str of the name of the attribute. For example,
            it can be "ner_type", which means the attribute ner_type of
            the entry will be treated as tags.

    Returns:
        BIO tag sequence in string format.

    """
    tag = bio_tagging(pack, instance, tagging_unit, entry_type, attribute)
    tag = [post_edit(x) for x in tag]
    return tag


def write_tokens_to_file(
    pred_pack: DataPack,
    refer_pack: DataPack,
    refer_request: Dict,
    tagging_unit: Type[Annotation],
    entry_type: Type[Annotation],
    attribute: str,
    output_file: str,
):
    r"""Write prediction results into files, along with reference
    labels, for performance evaluation.

    Args:
        pred_pack (DataPack): The predicated datapack.

        refer_pack (DataPack): The reference datapack.

        refer_request (Dict): Reference request.

        tagging_unit (Type[Annotation]): The type of tagging unit that entry
            tag should align to. For example, it can be Token, which means
            returned tags should aligned to tokens in one sentence.

        entry_type (Type[Annotation]): The type of entry that contains tags. For
            example, it can be EntityMethion, which means tags comes from the
            EntityMention of one sentence. Note that the number of EntityMention
            can be different from the number of Token. That is why we need to
            use BIO tagging to aglin them.

        attribute (str): A str of the name of the attribute. For example,
            it can be "ner_type", which means the attribute ner_type of the
            entry will be treated as tags.

        output_file (str): The path where results write.

    """
    opened_file = open(output_file, "a+")
    for refer_data, pred_sent, refer_sent in zip(
        refer_pack.get_data(**refer_request),
        pred_pack.get(Sentence),
        refer_pack.get(Sentence),
    ):

        refer_tag = get_tag(
            refer_pack, refer_sent, tagging_unit, entry_type, attribute
        )
        pred_tag = get_tag(
            pred_pack, pred_sent, tagging_unit, entry_type, attribute
        )

        words = refer_data["Token"]["text"]

        for i, (word, tgt, pred) in enumerate(
            zip(words, refer_tag, pred_tag), 1
        ):
            opened_file.write("%d %s %s %s\n" % (i, word, tgt, pred))
        opened_file.write("\n")
    opened_file.close()
