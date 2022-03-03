#  Copyright 2022 The Forte Authors. All Rights Reserved.
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

import functools
from forte.utils.utils import get_class

__all__ = ["EntryTypeGenerator"]


def _get_type_attributes():
    # TODO: implement get type_attributes using entry class hierachy
    type_attributes = {}

    return type_attributes


def _get_entry_attribute_by_class(input_entry_class_name: str):
    """Get type attributes by class name. input_entry_class_name should be
    an object class dotted name path. This method is actually relies on type annotation.
    """

    class_ = get_class(input_entry_class_name)
    try:
        return list(class_.__annotations__.keys())
    except AttributeError:
        return []


class EntryTypeGenerator:
    """
    The base class of entry type generator.
    """

    @staticmethod
    @functools.lru_cache(1)
    def get_type_attributes():
        """
        While initializing the new data structure, we need a list of all the valid annotations,
        links and groups. For example, for annotations, we want to have a list of types as strings,
        ["Token", "Subword", "Sentence", ...], including all the types defined in ft.onto and
        ftx.onto. Please notice that users could also create their own types of data which inherit
        "forte.data.ontology.core.Entry".
        For each type, we want to obtain all the attributes. For example, for sentence we want to
        get a list of ["speaker", "part_id", "sentiment", "classification", "classifications"]. We
        want to get the attributes for each type of entry as a dictionary. For example:

        .. code-block:: python

            type_attributes = {
                "Token": ["pos", "ud_xpos", "lemma", "chunk", "ner", "sense",
                        "is_root", "ud_features", "ud_misc"],
                "Document": ["document_class", "sentiment", "classifications"],
                "Sentence": ["speaker", "part_id", "sentiment",
                            "classification", "classifications"],
            }

        """
        return _get_type_attributes()

    @staticmethod
    def get_entry_attributes_by_class(input_entry_class_name: str):
        """Get type attributes by class name.

        Args:
            input_entry_class_name: An object class dotted name path.

        Returns:
             A list of attributes corresponding to the input class.

        For example, for Sentence we want to get a list of
        ["speaker", "part_id", "sentiment", "classification", "classifications"].
        The solution looks like the following:

        ... code-block::python

            # input can be a string
            entry_name = "ft.onto.base_ontology.Sentence"

            # function signature
            def get_entry_attributes_by_class(input_entry_class: str):

            # return
            ["speaker", "part_id", "sentiment", "classification", "classifications"]

        """
        return _get_entry_attribute_by_class(input_entry_class_name)
