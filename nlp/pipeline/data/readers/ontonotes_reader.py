"""
The reader that reads Ontonotes ner_data into Datapacks.
"""
import os
import codecs
from typing import DefaultDict, List, Optional, Iterator, Tuple
from collections import defaultdict
from nlp.pipeline.data.readers.file_reader import MonoFileReader
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.ontonotes_ontology import OntonotesOntology


class OntonotesReader(MonoFileReader):
    """:class:`OntonotesReader` is designed to read in the English OntoNotes
    v5.0 data in the format used by the CoNLL 2011/2012 shared tasks. To use
    this Reader, you must follow the instructions provided `here (v12 release):
    <http://cemantix.org/data/ontonotes.html>`_:, which will allow you to
    download the CoNLL style annotations for the OntoNotes v5.0 release
    – LDC2013T19.tgz obtained from LDC.

    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``dataset_iterator()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``dataset_iterator()`` returns a list.
    """
    def __init__(self, lazy: bool = True):
        super().__init__(lazy)
        self.ontology = OntonotesOntology

    @staticmethod
    def dataset_path_iterator(dir_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory containing
        CoNLL-formatted files.
        """
        for root, _, files in os.walk(dir_path):
            for data_file in files:
                if data_file.endswith("gold_conll"):
                    yield os.path.join(root, data_file)

    def _read_document(self, file_path: str) -> DataPack:
        doc = codecs.open(file_path, "r", encoding="utf8")

        text = ""
        offset = 0
        has_rows = False

        speaker = part_id = document_id = None
        sentence_begin = 0

        # auxiliary structures
        current_entity_mention: Optional[Tuple[int, str]] = None
        verbal_predicates: List[str] = []

        current_pred_arg: List[Optional[Tuple[str, str]]] = []
        verbal_pred_args: List[List[Tuple[str, str]]] = []

        groups: DefaultDict[int, List[str]] = defaultdict(list)
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
                kwargs_i = {"pos_tag": pos_tag, "sense": word_sense}
                token = self.ontology.Token(
                    self.component_name, word_begin, word_end
                )
                token.set_fields(**kwargs_i)
                self.current_datapack.add_entry(token)

                # add entity mentions
                current_entity_mention = self._process_entity_annotations(
                    entity_label, word_begin, word_end, current_entity_mention
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
                    pred_mention = self.ontology.PredicateMention(
                        self.component_name, word_begin, word_end
                    )
                    pred_mention.set_fields(**kwargs_i)
                    pred_mention_id = self.current_datapack.add_entry(
                        pred_mention
                    )

                    if word_is_verbal_predicate:
                        verbal_predicates.append(pred_mention_id)

                if not verbal_pred_args:
                    current_pred_arg = [None for _ in pred_labels]
                    verbal_pred_args = [[] for _ in pred_labels]

                # add predicate arguments
                self._process_pred_annotations(
                    conll_components[11:-1],
                    word_begin,
                    word_end,
                    current_pred_arg,
                    verbal_pred_args,
                )

                # add coreference mentions
                self._process_coref_annotations(
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
                            "parent": predicate,
                            "child": arg[0],
                        }
                        link = self.ontology.PredicateLink(self.component_name)
                        link.set_fields(**kwargs_i)
                        self.current_datapack.add_entry(link)

                verbal_predicates = []
                current_pred_arg = []
                verbal_pred_args = []

                # add sentence

                kwargs_i = {"speaker": speaker, "part_id": part_id}
                sent = self.ontology.Sentence(
                    self.component_name, sentence_begin, offset - 1
                )
                sent.set_fields(**kwargs_i)
                self.current_datapack.add_entry(sent)

                sentence_begin = offset

                has_rows = False

        # group the coreference mentions in the whole document
        for group_id, mention_list in groups.items():
            kwargs_i = {"coref_type": group_id}
            group = self.ontology.CoreferenceGroup(self.component_name)
            group.set_fields(**kwargs_i)
            group.add_members(mention_list)
            self.current_datapack.add_entry(group)

        kwargs_i = {"doc_id": document_id}
        self.current_datapack.set_meta(**kwargs_i)
        self.current_datapack.text = text

        doc.close()
        return self.current_datapack

    def _process_entity_annotations(self,
                                    label: str,
                                    word_begin: int,
                                    word_end: int,
                                    current_entity_mention: Tuple[int, str],
                                    ) -> Tuple[int, str]:

        ner_type = label.strip("()*")

        if "(" in label:
            # Entering into a span for a particular ner.
            current_entity_mention = (word_begin, ner_type)
        if ")" in label:
            # Exiting a span, add and then reset the current span.
            kwargs_i = {"ner_type": current_entity_mention[1]}
            entity = self.ontology.EntityMention(
                self.component_name, current_entity_mention[0], word_end
            )
            entity.set_fields(**kwargs_i)
            self.current_datapack.add_entry(entity)

            current_entity_mention = None

        return current_entity_mention

    def _process_pred_annotations(
            self,
            labels: List[str],
            word_begin: int,
            word_end: int,
            current_pred_arg: List[Optional[Tuple[int, str]]],
            verbal_pred_args: List[List[Tuple[str, str]]],
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
                arg_begin = current_pred_arg[label_index][0]
                arg_type = current_pred_arg[label_index][1]

                pred_arg = self.ontology.PredicateArgument(
                    self.component_name, arg_begin, word_end
                )
                pred_arg_id = self.current_datapack.add_entry(pred_arg)

                verbal_pred_args[label_index].append((pred_arg_id, arg_type))
                current_pred_arg[label_index] = None

    def _process_coref_annotations(
            self,
            label: str,
            word_begin: int,
            word_end: int,
            coref_stacks: DefaultDict[int, List[int]],
            groups: DefaultDict[int, List[str]],
    ) -> None:

        if label == "-":
            return
        for segment in label.split("|"):
            # The conll representation of coref spans allows spans to overlap.
            if segment[0] == "(":
                if segment[-1] == ")":
                    # The span begins and ends at this word (single word span).
                    group_id = int(segment[1:-1])

                    coref_mention = self.ontology.CoreferenceMention(
                        self.component_name, word_begin, word_end
                    )
                    coref_mention_id = self.current_datapack.add_entry(
                        coref_mention
                    )

                    groups[group_id].append(coref_mention_id)
                else:
                    # The span is starting, so we record the index of the word.
                    group_id = int(segment[1:])
                    coref_stacks[group_id].append(word_begin)
            else:
                # The span for this id is ending, but not start at this word.
                group_id = int(segment[:-1])
                start = coref_stacks[group_id].pop()
                coref_mention = self.ontology.CoreferenceMention(
                    self.component_name, start, word_end
                )
                coref_mention_id = self.current_datapack.add_entry(
                    coref_mention
                )

                groups[group_id].append(coref_mention_id)

    def _record_fields(self):
        self.current_datapack.record_fields(
            ["speaker", "part_id", "span"],
            self.component_name,
            self.ontology.Sentence.__name__,
        )
        self.current_datapack.record_fields(
            ["sense", "pos_tag", "span"],
            self.component_name,
            self.ontology.Token.__name__,
        )
        self.current_datapack.record_fields(
            ["ner_type", "span"],
            self.component_name,
            self.ontology.EntityMention.__name__,
        )
        self.current_datapack.record_fields(
            ["pred_lemma", "pred_type", "span", "framenet_id"],
            self.component_name,
            self.ontology.PredicateMention.__name__,
        )
        self.current_datapack.record_fields(
            ["span"],
            self.component_name,
            self.ontology.PredicateArgument.__name__,
        )
        self.current_datapack.record_fields(
            ["parent", "child", "arg_type"],
            self.component_name,
            self.ontology.PredicateLink.__name__,
        )
        self.current_datapack.record_fields(
            ["span"],
            self.component_name,
            self.ontology.CoreferenceMention.__name__,
        )
        self.current_datapack.record_fields(
            ["coref_type", "members"],
            self.component_name,
            self.ontology.CoreferenceGroup.__name__,
        )
