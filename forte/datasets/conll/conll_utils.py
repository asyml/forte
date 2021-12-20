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
from typing import List, Dict, Optional, Type, Tuple, Union, Callable

from forte.data.data_pack import DataPack
from forte.data.ontology import Annotation
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


def bio_tagging(
    pack: DataPack,
    tagging_unit_type: Type[Annotation],
    entry_type: Type[Annotation],
    attribute: Union[Callable[[Annotation], str], str],
    context: Optional[Annotation] = None,
) -> List[Tuple[Optional[str], str]]:
    """This utility function use BIO tagging method to convert tags
    of "instance_entry" into the same length as "instance_tagging_unit". Both
    element from "instance_entry" and "instance_tagging_unit" should Annotation
    type. This function uses their position information to
    determine B, I, O tagging for the entry on each tagging_unit element.

    Args:
        pack (DataPack): The datapack that contains the current
            instance.

        tagging_unit_type (Annotation): The type of tagging unit that entry
            tag should align to. For example, it can be Token, which means
            returned tags should aligned to tokens in one sentence.

        entry_type (Annotation): The type of entry that contains tags. For
            example, it can be EntityMention, which means tags comes from the
            EntityMention of one sentence. Note that the number of EntityMention
            can be different from the number of Token. That is why we need to
            use BIO tagging to align them.

        attribute (Union[Callable[[Annotation], str], str]): A function to
            get the tags via the attribute of an entry. Or a str of the name
            of the attribute. For example, it can be "ner_type", which means
            the attribute ner_type of the entry will be treated as tags.

        context (Annotation): The instance from which the
            extractor will extractor feature. For example, an instance of
            Sentence type, which mean the tagging sequence comes from
            one sentence. If None, then the whole data pack will be used.

    Returns:
        A list of the type List[Tuple[Optional[str], str]]. For example,
        [(None, "O"), (LOC, "B"), (LOC, "I"), (None, "O"),
         (None, "O"), (PER, "B"), (None, "O")]
    """

    # Tokens in the sentence.
    tagging_units: List[Annotation] = list(pack.get(tagging_unit_type, context))
    # All mentions in the sentence.
    instance_entry: List[Annotation] = list(pack.get(entry_type, context))

    tagged: List[Tuple[Optional[str], str]] = []
    unit_id = 0
    entry_id = 0
    while unit_id < len(tagging_units) or entry_id < len(instance_entry):
        if entry_id == len(instance_entry):
            tagged.append((None, "O"))
            unit_id += 1
            continue

        is_start = True
        for unit in pack.get(tagging_unit_type, instance_entry[entry_id]):
            while (
                unit_id < len(tagging_units)
                and tagging_units[unit_id].begin != unit.begin
                and tagging_units[unit_id].end != unit.end
            ):
                tagged.append((None, "O"))
                unit_id += 1

            if is_start:
                location = "B"
                is_start = False
            else:
                location = "I"

            unit_id += 1
            if callable(attribute):
                tagged.append((attribute(instance_entry[entry_id]), location))
            else:
                tagged.append(
                    (getattr(instance_entry[entry_id], attribute), location)
                )
        entry_id += 1
    return tagged


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
    tag = bio_tagging(pack, tagging_unit, entry_type, attribute, instance)
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
    with open(output_file, "a+", encoding="utf-8") as opened_file:
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
