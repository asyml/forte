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
The reader that reads CoNLL-U Standard Universal Dependency Format -
https://universaldependencies.org/docs/format.html
into data_pack format
"""
from typing import Iterator, Dict, Tuple, Any

from ft.onto.base_ontology import Document, Sentence, Token, Dependency, \
    EnhancedDependency
from forte.data.io_utils import dataset_path_iterator
from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader

__all__ = [
    "ConllUDReader"
]


class ConllUDReader(PackReader):
    """
    :class:`conllUReader` is designed to read in the Universal Dependencies
    2.4 dataset.
    """

    def _cache_key_function(self, data_pack: Any) -> str:
        if data_pack.meta.doc_id is None:
            raise ValueError("data_pack does not have a document id")
        return data_pack.meta.doc_id

    def _collect(self, *args, **kwargs) -> Iterator[Any]:
        # pylint: disable = unused-argument
        """
        Iterator over conll files in the data_source

        Args:
            args: args[0] is the directory to the conllu files.
            kwargs:

        Returns: data packs obtained from each document from each conllu file.
        """
        conll_dir_path = args[0]

        file_paths = dataset_path_iterator(conll_dir_path, "conllu")
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf8") as file:
                lines = file.readlines()
                doc_lines = []

                for i, line in enumerate(lines):
                    # previous document ends
                    doc_lines.append(line)
                    if i == len(lines) - 1 or \
                            lines[i + 1].strip().startswith("# newdoc"):
                        yield doc_lines
                        doc_lines = []

    def _parse_pack(self, doc_lines) -> Iterator[DataPack]:
        token_comp_fields = ["id", "form", "lemma", "pos",
                             "ud_xpos", "ud_features", "head", "label",
                             "enhanced_dependency_relations", "ud_misc"]

        token_multi_fields = ["ud_features", "ud_misc",
                              "enhanced_dependency_relations"]

        token_feature_fields = ["ud_features", "ud_misc"]

        token_entry_fields = ["lemma", "pos", "ud_xpos",
                              "ud_features", "ud_misc"]

        data_pack: DataPack = DataPack()
        doc_sent_begin: int = 0
        doc_num_sent: int = 0
        doc_text: str = ''
        doc_offset: int = 0
        doc_id: str

        sent_text: str
        sent_tokens: Dict[str, Tuple[Dict[str, Any], Token]] = {}

        for line in doc_lines:
            line = line.strip()
            line_comps = line.split()

            if line.startswith("# newdoc"):
                doc_id = line.split("=")[1].strip()

            elif line.startswith("# sent"):
                sent_text = ''

            elif len(line_comps) > 0 and \
                    line_comps[0].strip().isdigit():
                # token
                token_comps: Dict[str, Any] = {}

                for index, key in enumerate(token_comp_fields):
                    token_comps[key] = str(line_comps[index])

                    if key in token_multi_fields:
                        values = str(token_comps[key]).split("|") \
                            if token_comps[key] != '_' else []
                        if key not in token_feature_fields:
                            token_comps[key] = values
                        else:
                            feature_lst = [elem.split('=', 1)
                                           for elem in values]
                            feature_dict = {elem[0]: elem[1]
                                            for elem in feature_lst}
                            token_comps[key] = feature_dict

                word: str = token_comps["form"]
                word_begin = doc_offset
                word_end = doc_offset + len(word)

                token: Token \
                    = Token(data_pack, word_begin, word_end)
                kwargs = {key: token_comps[key]
                          for key in token_entry_fields}

                # add token
                token.set_fields(**kwargs)
                data_pack.add_or_get_entry(token)

                sent_tokens[str(token_comps["id"])] = (token_comps, token)

                sent_text += word + " "
                doc_offset = word_end + 1

            elif line == "":
                # sentence ends
                sent_text = sent_text.strip()
                doc_text += ' ' + sent_text

                # add dependencies for a sentence when all the tokens have been
                # added
                for token_id in sent_tokens:
                    token_comps, token = sent_tokens[token_id]

                    # add primary dependency
                    label = token_comps["label"]
                    if label == "root":
                        token.is_root = True
                    else:
                        token.is_root = False
                        head = sent_tokens[token_comps["head"]][1]
                        dependency = Dependency(data_pack, head, token)
                        dependency.dep_label = label
                        data_pack.add_or_get_entry(dependency)

                    # add enhanced dependencies
                    for dep in token_comps["enhanced_dependency_relations"]:
                        head_id, label = dep.split(":", 1)
                        if label != "root":
                            head = sent_tokens[head_id][1]
                            dependency = EnhancedDependency(data_pack, head,
                                                            token)
                            dependency.dep_label = label
                            data_pack.add_or_get_entry(dependency)

                # add sentence
                sent = Sentence(data_pack, doc_sent_begin, doc_offset - 1)
                data_pack.add_or_get_entry(sent)

                doc_sent_begin = doc_offset
                doc_num_sent += 1

        # add doc to data_pack
        document = Document(data_pack, 0, len(doc_text))
        data_pack.add_or_get_entry(document)
        data_pack.meta.doc_id = doc_id
        data_pack.set_text(doc_text.strip())

        yield data_pack
