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
    # TODO: implement get type_attributes using entry class hierachy.
    type_attributes = {}

    return type_attributes


class EntryTypeGenerator:
    """The base class of entry type generator."""

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
    def get_entry_attributes_by_class(input_entry_class_name: str) -> list:
        """Get type attributes by class name. `input_entry_class_name` should be
        a fully qualified name of an entry class.

        The `dataclass` module<https://docs.python.org/3/library/dataclasses.html> can add
        generated special methods to user-defined classes. There is an in-built function
        called `__dataclass_fields__` that is called on the class object, and it returns
        all the fields the class contains.

        .. note::

            This function is only applicable to classes decorated as Python
            `dataclass` since it relies on the `__dataclass_fields__` to find out the attributes.


        Args:
            input_entry_class_name: A fully qualified name of an entry class.

        Returns:
            A list of attributes corresponding to the input class.

        For example, for Sentence we want to get a list of
        ["speaker", "part_id", "sentiment", "classification", "classifications"].
        The solution looks like the following:

        .. code-block:: python

            # input can be a string
            entry_name = "ft.onto.base_ontology.Sentence"

            # function signature
            get_entry_attributes_by_class(entry_name)

            # return
            # ["speaker", "part_id", "sentiment", "classification", "classifications"]

        """
        class_ = get_class(input_entry_class_name)
        try:
            return list(class_.__dataclass_fields__.keys())
        except AttributeError:
            return []
