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
Utility functions related to processors.
"""

__all__ = [
    "parse_allennlp_srl_tags"
]

from typing import Dict, List
from forte.data.span import Span
from ft.onto.base_ontology import Token


def parse_allennlp_srl_tags(tokens: List[Token], tags: List[str]):
    r"""Parse the tag list of a specific verb output by
    AllenNLP SRL processor.

    Args:
        tokens (list): A list of Tokens.
        tags (list): a list of semantic role lables.

    Returns:
         the span of the verb and
         its semantic role arguments.
    """
    pred_span = None
    arguments = []
    begin, end, prev_argument = None, None, ''
    for i, item in enumerate(tags):
        argument = '-'.join(item.split('-')[1:])
        if prev_argument not in ('', argument):
            if prev_argument == 'V':
                pred_span = Span(tokens[begin].begin, tokens[end].end)
            else:
                arg_span = Span(tokens[begin].begin, tokens[end].end)
                arguments.append((arg_span, prev_argument))
        prev_argument = argument
        if item.startswith('B-'):
            begin = i
            end = i
        if item.startswith('I-'):
            end = i
    return pred_span, arguments
