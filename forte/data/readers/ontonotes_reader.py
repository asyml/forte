"""
The reader that reads Ontonotes data into Datapacks.
"""
import os
from collections import defaultdict
from typing import (DefaultDict, List, Optional, Tuple,
                    Dict, Any, Iterator)

from forte.data.data_pack import DataPack
from forte.data.io_utils import dataset_path_iterator
from forte.data.ontology import ontonotes_ontology
from forte.data.ontology.base_ontology import (
    PredicateMention, PredicateArgument, CoreferenceMention)
from forte.data.readers.base_reader import PackReader

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

    """

    def __init__(self):
        super().__init__()
        self._ontology = ontonotes_ontology
        self.define_output_info()

    def define_output_info(self):
        return {
            self._ontology.Document: [],
            self._ontology.Sentence: ["speaker", "part_id"],
            self._ontology.Token: ["sense", "pos_tag"],
            self._ontology.EntityMention: ["ner_type"],
            self._ontology.PredicateMention:
                ["pred_lemma", "pred_type", "framenet_id"],
            self._ontology.PredicateArgument: [],
            self._ontology.PredicateLink: ["parent", "child", "arg_type"],
            self._ontology.CoreferenceMention: [],
            self._ontology.CoreferenceGroup: ["coref_type", "members"]
        }

    # pylint: disable=no-self-use
    def _collect(self, conll_directory: str) -> Iterator[Any]:  # type: ignore
        """
        Iterator over *.gold_conll files in the data_source

        Args:
            conll_directory:  path to the directory containing the files.

        Returns: Iterator over files with gold_conll path.

        """
        return dataset_path_iterator(conll_directory, "gold_conll")

    # pylint: disable=no-self-use
    def _cache_key_function(self, conll_file: str) -> str:
        return os.path.basename(conll_file)

    def parse_pack(self, file_path: str) -> DataPack:
        pack = DataPack()

        with open(file_path, encoding="utf8") as doc:
            text = ""
            offset = 0
            has_rows = False

            speaker = part_id = document_id = None
            sentence_begin = 0

            # auxiliary structures
            current_entity_mention: Optional[Tuple[int, str]] = None
            verbal_predicates: List[PredicateMention] = []

            current_pred_arg: List[Optional[Tuple[int, str]]] = []
            verbal_pred_args: List[List[Tuple[PredicateArgument, str]]] = []

            groups: DefaultDict[int, List[CoreferenceMention]] = \
                defaultdict(list)
            coref_stacks: DefaultDict[int, List[int]] = defaultdict(list)

            for line in doc:
                line = line.strip()

                if line.startswith("#end document"):
                    break

                if line != "" and not line.startswith("#"):
                    conll_components = line.split()
                    document_id = conll_components[0]
                    part_id = int(conll_components[1])
                    word = conll_components[3]
                    pos_tag = conll_components[4]
                    lemmatised_word = conll_components[6]
                    framenet_id = conll_components[7]
                    word_sense = conll_components[8]
                    speaker = conll_components[9]
                    entity_label = conll_components[10]
                    pred_labels = conll_components[11:-1]

                    word_begin = offset
                    word_end = offset + len(word)

                    # add tokens
                    kwargs_i: Dict[str, Any] = {"pos_tag": pos_tag,
                                                "sense": word_sense}
                    token = self._ontology.Token(  # type: ignore
                        pack, word_begin, word_end
                    )
                    token.set_fields(**kwargs_i)
                    pack.add_or_get_entry(token)

                    # add entity mentions
                    current_entity_mention = self._process_entity_annotations(
                        pack, entity_label, word_begin, word_end,
                        current_entity_mention
                    )

                    # add predicate mentions
                    if lemmatised_word != "-":
                        word_is_verbal_predicate = any(
                            ["(V" in x for x in pred_labels]
                        )
                        kwargs_i = {
                            "framenet_id": framenet_id,
                            "pred_lemma": lemmatised_word,
                            "pred_type": "verb" if word_is_verbal_predicate
                            else "other"
                        }
                        pred_mention = \
                            self._ontology.PredicateMention(  # type: ignore
                                pack, word_begin, word_end
                            )
                        pred_mention.set_fields(**kwargs_i)
                        pred_mention = pack.add_or_get_entry(
                            pred_mention
                        )

                        if word_is_verbal_predicate:
                            verbal_predicates.append(pred_mention)

                    if not verbal_pred_args:
                        current_pred_arg = [None for _ in pred_labels]
                        verbal_pred_args = [[] for _ in pred_labels]

                    # add predicate arguments
                    self._process_pred_annotations(
                        pack,
                        conll_components[11:-1],
                        word_begin,
                        word_end,
                        current_pred_arg,
                        verbal_pred_args,
                    )

                    # add coreference mentions
                    self._process_coref_annotations(
                        pack,
                        conll_components[-1],
                        word_begin,
                        word_end,
                        coref_stacks,
                        groups,
                    )

                    text += word + " "
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
                            link = self._ontology.PredicateLink(  # type: ignore
                                pack, predicate, arg[0]
                            )
                            link.set_fields(**kwargs_i)
                            pack.add_or_get_entry(link)

                    verbal_predicates = []
                    current_pred_arg = []
                    verbal_pred_args = []

                    # add sentence

                    kwargs_i = {"speaker": speaker, "part_id": part_id}
                    sent = self._ontology.Sentence(  # type: ignore
                        pack, sentence_begin, offset - 1
                    )
                    sent.set_fields(**kwargs_i)
                    pack.add_or_get_entry(sent)

                    sentence_begin = offset

                    has_rows = False

            # group the coreference mentions in the whole document
            for group_id, mention_list in groups.items():
                kwargs_i = {"coref_type": group_id}
                group = self._ontology.CoreferenceGroup(  # type: ignore
                    pack)
                group.set_fields(**kwargs_i)
                group.add_members(mention_list)
                pack.add_or_get_entry(group)

            document = self._ontology.Document(  # type: ignore
                pack, 0, len(text))
            pack.add_or_get_entry(document)

            kwargs_i = {"doc_id": document_id}
            pack.set_meta(**kwargs_i)
            pack.set_text(text, replace_func=self.text_replace_operation)

        return pack

    def _process_entity_annotations(
            self,
            pack: DataPack,
            label: str,
            word_begin: int,
            word_end: int,
            current_entity_mention: Optional[Tuple[int, str]],
    ) -> Optional[Tuple[int, str]]:

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
            entity = self._ontology.EntityMention(  # type: ignore
                pack, current_entity_mention[0], word_end
            )
            entity.set_fields(**kwargs_i)
            pack.add_or_get_entry(entity)

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

            arg_type = label.strip("()*")
            if arg_type == "V":
                continue

            if "(" in label:
                # Entering into a span
                current_pred_arg[label_index] = (word_begin, arg_type)
            if ")" in label:
                # Exiting a span
                if current_pred_arg[label_index] is None:
                    raise ValueError(
                        "current_pred_arg is None when meet right blanket.")

                arg_begin = current_pred_arg[label_index][0]  # type: ignore
                arg_type = current_pred_arg[label_index][1]  # type: ignore

                pred_arg = self._ontology.PredicateArgument(  # type: ignore
                    pack, arg_begin, word_end
                )
                pred_arg = pack.add_or_get_entry(pred_arg)

                verbal_pred_args[label_index].append((pred_arg, arg_type))
                current_pred_arg[label_index] = None

    def _process_coref_annotations(
            self,
            pack: DataPack,
            label: str,
            word_begin: int,
            word_end: int,
            coref_stacks: DefaultDict[int, List[int]],
            groups: DefaultDict[int, List[CoreferenceMention]],
    ) -> None:

        if label == "-":
            return
        for segment in label.split("|"):
            # The conll representation of coref spans allows spans to overlap.
            if segment[0] == "(":
                if segment[-1] == ")":
                    # The span begins and ends at this word (single word span).
                    group_id = int(segment[1:-1])

                    coref_mention = \
                        self._ontology.CoreferenceMention(  # type: ignore
                            pack, word_begin, word_end
                        )
                    coref_mention = pack.add_or_get_entry(
                        coref_mention
                    )

                    groups[group_id].append(coref_mention)
                else:
                    # The span is starting, so we record the index of the word.
                    group_id = int(segment[1:])
                    coref_stacks[group_id].append(word_begin)
            else:
                # The span for this id is ending, but not start at this word.
                group_id = int(segment[:-1])
                start = coref_stacks[group_id].pop()
                coref_mention = \
                    self._ontology.CoreferenceMention(  # type: ignore
                        pack, start, word_end
                    )
                coref_mention = pack.add_or_get_entry(
                    coref_mention
                )

                groups[group_id].append(coref_mention)
