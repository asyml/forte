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
The reader that reads Ontonotes data into Datapacks.
"""
import os
from collections import defaultdict
from typing import (Any, DefaultDict, Iterator, List, NamedTuple, Optional,
                    Set, Tuple)

from forte.data.data_pack import DataPack
from forte.data.io_utils import dataset_path_iterator
from forte.data.readers.base_reader import PackReader
from ft.onto.base_ontology import (
    CoreferenceGroup, Document, EntityMention, PredicateArgument, PredicateLink,
    PredicateMention, Sentence, Token)

__all__ = [
    "OntonotesReader",
]


class OntonotesReader(PackReader):
    """:class:`OntonotesReader` is designed to read in the English OntoNotes
    v5.0 data in the datasets used by the CoNLL 2011/2012 shared tasks. To use
    this Reader, you must follow the instructions provided `here (v12 release):
    <http://cemantix.org/data/ontonotes.html>`_:, which will allow you to
    download the CoNLL style annotations for the OntoNotes v5.0 release
    – LDC2013T19.tgz obtained from LDC.

    Args:
        column_format: A list of strings indicating which field each column in a
            line corresponds to. The length of the list should be equal to the
            number of columns in the files to be read. Available field types
            include:

            - ``"document_id"``
            - ``"part_number"``
            - ``"word"``
            - ``"pos_tag"``
            - ``"lemmatised_word"``
            - ``"framenet_id"``
            - ``"word_sense"``
            - ``"speaker"``
            - ``"entity_label"``
            - ``"coreference"``
            - ``"*predicate_labels"``

            Field types marked with ``*`` indicate a variable-column field: it
            could span multiple fields. Only one such field is allowed in the
            format specification.

            If a column should be ignored, fill in `None` at the corresponding
            position.
    """

    class ParsedFields(NamedTuple):
        word: str
        predicate_labels: List[str]
        document_id: Optional[str] = None
        part_number: Optional[str] = None
        pos_tag: Optional[str] = None
        lemmatised_word: Optional[str] = None
        framenet_id: Optional[str] = None
        word_sense: Optional[str] = None
        speaker: Optional[str] = None
        entity_label: Optional[str] = None
        coreference: Optional[str] = None

    _DEFAULT_FORMAT = [
        "document_id", "part_number", None, "word", "pos_tag", None,
        "lemmatised_word", "framenet_id", "word_sense", "speaker",
        "entity_label", "*predicate_labels", "coreference",
    ]
    _STAR_FIELDS = {"predicate_labels"}
    _REQUIRED_FIELDS = ["word", "predicate_labels"]

    def __init__(self, column_format: Optional[List[Optional[str]]] = None):
        super().__init__()
        column_format = column_format or self._DEFAULT_FORMAT
        # Validate column format.
        seen_fields: Set[str] = set()
        self._column_format: List[Optional[str]] = []
        self._star_pos = None
        for idx, field in enumerate(column_format):
            if field is None:
                self._column_format.append(None)
                continue
            if field.startswith("*"):
                if self._star_pos is not None:
                    raise ValueError(f"Only one field can begin with '*'")
                field = field[1:]
                if field not in self._STAR_FIELDS:
                    raise ValueError(f"Field '{field}' cannot begin with '*'")
                self._star_pos = idx
            if field not in self.ParsedFields._fields:
                raise ValueError(f"Unsupported field type: '{field}'")
            if field in seen_fields:
                raise ValueError(f"Duplicate field type: '{field}'")
            seen_fields.add(field)
            self._column_format.append(field)
        # Sanity check: certain fields must be present in format.
        for field in self._REQUIRED_FIELDS:
            if field not in seen_fields:
                raise ValueError(f"'{field}' field is required")

    def _collect(self, conll_directory: str) -> Iterator[Any]:  # type: ignore
        """
        Iterator over *.gold_conll files in the data_source

        Args:
            conll_directory:  path to the directory containing the files.

        Returns: Iterator over files with gold_conll path.

        """
        return dataset_path_iterator(conll_directory, "gold_conll")

    def _cache_key_function(self, conll_file: str) -> str:
        return os.path.basename(conll_file)

    def _parse_line(self, line: str) -> 'ParsedFields':
        parts = line.split()
        fields = {}
        if self._star_pos is not None:
            l = self._star_pos
            r = len(parts) - (len(self._column_format) - self._star_pos - 1)
            parts = parts[:l] + [parts[l:r]] + parts[r:]  # type: ignore
        for field, part in zip(self._column_format, parts):
            if field is not None:
                fields[field] = part
        return self.ParsedFields(**fields)  # type: ignore

    def _parse_pack(self, file_path: str) -> Iterator[DataPack]:
        pack = DataPack()

        with open(file_path, encoding="utf8") as doc:
            words = []
            offset = 0
            has_rows = False

            speaker = part_id = document_id = None
            sentence_begin = 0

            # auxiliary structures
            current_entity_mention: Optional[Tuple[int, str]] = None
            verbal_predicates: List[PredicateMention] = []

            current_pred_arg: List[Optional[Tuple[int, str]]] = []
            verbal_pred_args: List[List[Tuple[PredicateArgument, str]]] = []

            groups: DefaultDict[int, List[EntityMention]] = defaultdict(list)
            coref_stacks: DefaultDict[int, List[int]] = defaultdict(list)

            for line in doc:
                line = line.strip()

                if line.startswith("#end document"):
                    break

                if line != "" and not line.startswith("#"):
                    fields = self._parse_line(line)
                    speaker = fields.speaker
                    if fields.part_number is not None:
                        part_id = int(fields.part_number)
                    document_id = fields.document_id

                    assert fields.word is not None
                    word_begin = offset
                    word_end = offset + len(fields.word)

                    # add tokens
                    token = Token(pack, word_begin, word_end)
                    if fields.pos_tag is not None:
                        token.set_fields(pos=fields.pos_tag)
                    if fields.word_sense is not None:
                        token.set_fields(sense=fields.word_sense)
                    pack.add_entry(token)

                    # add entity mentions
                    current_entity_mention = self._process_entity_annotations(
                        pack, fields.entity_label, word_begin, word_end,
                        current_entity_mention,
                    )

                    # add predicate mentions
                    if (fields.lemmatised_word is not None and
                            fields.lemmatised_word != "-"):
                        word_is_verbal_predicate = any(
                            "(V" in x for x in fields.predicate_labels)
                        kwargs_i = {
                            "predicate_lemma": fields.lemmatised_word,
                            "is_verb": word_is_verbal_predicate
                        }
                        pred_mention = PredicateMention(
                            pack, word_begin, word_end)
                        pred_mention.set_fields(**kwargs_i)
                        if fields.framenet_id is not None:
                            pred_mention.set_fields(
                                framenet_id=fields.framenet_id)
                        pack.add_entry(pred_mention)

                        if word_is_verbal_predicate:
                            verbal_predicates.append(pred_mention)

                    if not verbal_pred_args:
                        current_pred_arg = [None] * len(fields.predicate_labels)
                        verbal_pred_args = [[] for _ in fields.predicate_labels]

                    # add predicate arguments
                    self._process_pred_annotations(
                        pack,
                        fields.predicate_labels,
                        word_begin,
                        word_end,
                        current_pred_arg,
                        verbal_pred_args,
                    )

                    # add coreference mentions
                    self._process_coref_annotations(
                        pack,
                        fields.coreference,
                        word_begin,
                        word_end,
                        coref_stacks,
                        groups,
                    )

                    words.append(fields.word)
                    offset = word_end + 1
                    has_rows = True

                else:
                    if not has_rows:
                        continue

                    # add predicate links in the sentence
                    for predicate, pred_arg in zip(verbal_predicates,
                                                   verbal_pred_args):
                        for arg in pred_arg:
                            kwargs_i = {
                                "arg_type": arg[1],
                            }
                            link = PredicateLink(pack, predicate, arg[0])
                            link.set_fields(**kwargs_i)
                            pack.add_entry(link)

                    verbal_predicates = []
                    current_pred_arg = []
                    verbal_pred_args = []

                    # add sentence

                    sent = Sentence(pack, sentence_begin, offset - 1)
                    if speaker is not None:
                        sent.set_fields(speaker=speaker)
                    if part_id is not None:
                        sent.set_fields(part_id=int(part_id))
                    pack.add_entry(sent)

                    sentence_begin = offset

                    has_rows = False

            # group the coreference mentions in the whole document
            for _, mention_list in groups.items():
                # kwargs_i = {"coref_type": group_id}
                group = CoreferenceGroup(pack)
                # group.set_fields(**kwargs_i)
                group.add_members(mention_list)
                pack.add_entry(group)

            text = " ".join(words)
            document = Document(pack, 0, len(text))
            pack.add_entry(document)

            if document_id is not None:
                pack.set_meta(doc_id=document_id)
            pack.set_text(text, replace_func=self.text_replace_operation)

        yield pack

    def _process_entity_annotations(
            self,
            pack: DataPack,
            label: Optional[str],
            word_begin: int,
            word_end: int,
            current_entity_mention: Optional[Tuple[int, str]],
    ) -> Optional[Tuple[int, str]]:
        if label is None:
            return None

        ner_type = label.strip("()*")

        if "(" in label:
            # Entering into a span for a particular ner.
            current_entity_mention = (word_begin, ner_type)
        if ")" in label:
            if current_entity_mention is None:
                raise ValueError(
                    "current_entity_mention is None when meet right blanket.")
            # Exiting a span, add and then reset the current span.
            kwargs_i = {"ner_type": current_entity_mention[1]}
            entity = EntityMention(pack, current_entity_mention[0], word_end)
            entity.set_fields(**kwargs_i)
            pack.add_entry(entity)

            current_entity_mention = None

        return current_entity_mention

    def _process_pred_annotations(
            self,
            pack: DataPack,
            labels: List[str],
            word_begin: int,
            word_end: int,
            current_pred_arg: List[Optional[Tuple[int, str]]],
            verbal_pred_args: List[List[Tuple[PredicateArgument, str]]],
    ) -> None:

        for label_index, label in enumerate(labels):

            if "(" in label:
                # Entering into a span
                arg_type = label.strip("()*")
                current_pred_arg[label_index] = (word_begin, arg_type)

            if ")" in label:
                # Exiting a span
                if current_pred_arg[label_index] is None:
                    raise ValueError(
                        "current_pred_arg is None when meet right blanket.")

                arg_begin = current_pred_arg[label_index][0]  # type: ignore
                arg_type = current_pred_arg[label_index][1]  # type: ignore

                if arg_type != "V":
                    pred_arg = PredicateArgument(pack, arg_begin, word_end)
                    pred_arg = pack.add_entry(pred_arg)

                    verbal_pred_args[label_index].append((pred_arg, arg_type))
                current_pred_arg[label_index] = None

    def _process_coref_annotations(
            self,
            pack: DataPack,
            label: Optional[str],
            word_begin: int,
            word_end: int,
            coref_stacks: DefaultDict[int, List[int]],
            groups: DefaultDict[int, List[EntityMention]],
    ) -> None:

        if label is None or label == "-":
            return
        for segment in label.split("|"):
            # The conll representation of coref spans allows spans to overlap.
            if segment[0] == "(":
                if segment[-1] == ")":
                    # The span begins and ends at this word (single word span).
                    group_id = int(segment[1:-1])

                    coref_mention = EntityMention(pack, word_begin, word_end)
                    coref_mention = pack.add_entry(coref_mention)

                    groups[group_id].append(coref_mention)
                else:
                    # The span is starting, so we record the index of the word.
                    group_id = int(segment[1:])
                    coref_stacks[group_id].append(word_begin)
            else:
                # The span for this id is ending, but not start at this word.
                group_id = int(segment[:-1])
                start = coref_stacks[group_id].pop()
                coref_mention = EntityMention(pack, start, word_end)
                coref_mention = pack.add_or_get_entry(coref_mention)

                groups[group_id].append(coref_mention)
