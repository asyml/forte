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

from typing import List, Optional

from ft.onto.base_ontology import LinkedMention


# class UMLSEntity:
#     """
#     A span based annotation for the UMLS Knowledge base.
#     """
#
#     def __init__(self,
#                  concept_id: str,
#                  canonical_name: str,
#                  aliases: List[str],
#                  types: List[str],
#                  definition: Optional[str] = None,
#                  *args, **kwargs):
#         self.aliases = aliases
#         self.concept_id = concept_id
#         self.canonical_name = canonical_name
#         self.types = types
#         self.definition = definition
#
#     def __repr__(self):
#         """
#         Borrowed from
#
#         """
#         rep = ""
#         num_aliases = len(self.aliases)
#         rep = rep + f"CUI: {self.concept_id}, Name: {self.canonical_name}\n"
#         rep = rep + f"Definition: {self.definition}\n"
#         rep = rep + f"TUI(s): {', '.join(self.types)}\n"
#         if num_aliases > 10:
#             rep = (
#                     rep
#                     + f"Aliases (abbreviated, total: "
#                     f"{num_aliases}): \n\t {', '.join(self.aliases[:10])}"
#             )
#         else:
#             rep = (
#                     rep + f"Aliases: (total: "
#                     f"{num_aliases}): \n\t {', '.join(self.aliases)}"
#             )
#         return rep


# class UMLSLinkedMention(LinkedMention):
#     """
#     A span based annotation :class:`LinkedMention`.
#     """




