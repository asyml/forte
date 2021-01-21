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
    "parse_allennlp_srl_results",
    "parse_allennlp_srl_tags"
]

from typing import Dict, List, Tuple, Any
from collections import defaultdict
from forte.data.span import Span
# from ft.onto.base_ontology import Token


def parse_allennlp_srl_results(
        results: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    r"""Convert SRL output into a dictionary
    of verbs and tags.

    Args:
        results (dict):
            the verb dictionary output by AllenNLP SRL

    Returns:
         a dictionary of verbs and tags
    """
    parsed_results: Dict[str, List[str]] = defaultdict(list)
    for verb_item in results:
        parsed_results['verbs'].append(verb_item['verb'])
        parsed_results['srl_tags'].append(
            ' '.join(verb_item['tags']))
    return parsed_results


def parse_allennlp_srl_tags(tags: str) -> \
        Tuple[Span, List[Tuple[Span, str]]]:
    r"""Parse the tag list of a specific verb output by
    AllenNLP SRL processor.

    Args:
        tags (str): a str of semantic role lables.

    Returns:
         the span of the verb and
         its semantic role arguments.
    """
    pred_span = None
    arguments = []
    begin, end, prev_argument = -1, -1, ''
    for i, item in enumerate(tags.split()):
        argument = '-'.join(item.split('-')[1:])
        if prev_argument not in ('', argument):
            if prev_argument == 'V':
                pred_span = Span(begin, end)
            else:
                arg_span = Span(begin, end)
                arguments.append((arg_span, prev_argument))
        prev_argument = argument
        if item.startswith('B-'):
            begin = i
            end = i
        if item.startswith('I-'):
            end = i
    if not pred_span:
        raise Exception('No verb detected in this sentence')
    return pred_span, arguments
