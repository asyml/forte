# Copyright 2021 The Forte Authors. All Rights Reserved.
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

from typing import Dict, List, Tuple, Any, Optional
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
    parsed_results['verbs'] = []
    parsed_results['srl_tags'] = []
    for verb_item in results:
        parsed_results['verbs'].append(verb_item['verb'])
        parsed_results['srl_tags'].append(
            ' '.join(verb_item['tags']))
    return parsed_results


def parse_allennlp_srl_tags(tags: str) -> \
        Tuple[Optional[Span], List[Tuple[Span, str]]]:
    r"""Parse the tag list of a specific verb output by
    AllenNLP SRL processor.

    Args:
        tags (str): a str of semantic role labels.

    Returns:
         the span of the verb and
         its semantic role arguments.
    """
    pred_span = None
    arguments = []
    begin, end, prev_argument = -1, -1, ''
    tags += ' O'
    for i, tag in enumerate(tags.split()):
        argument = '-'.join(tag.split('-')[1:])
        if tag[0] == 'O' or tag[0] == 'B' or \
            (tag[0] == 'I' and argument != prev_argument):
            if prev_argument == 'V':
                pred_span = Span(begin, end)
            elif prev_argument != '':
                arg_span = Span(begin, end)
                arguments.append((arg_span, prev_argument))
            begin = i
            end = i
            prev_argument = argument
        else:
            end = i

    return pred_span, arguments
