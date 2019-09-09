"""
The reader that reads CoNLL-U Standard Universal Dependency Format -
https://universaldependencies.org/docs/format.html
into data_pack format
"""
from typing import Iterator, Dict, Tuple, Any

from forte.data.io_utils import dataset_path_iterator
from forte.data.ontology.universal_dependency_ontology import \
    (Document, Sentence, DependencyToken, UniversalDependency)
from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader

__all__ = [
    "ConllUReader"
]


class ConllUReader(PackReader):
    """:class:`conllUReader` is designed to read in the Universal Dependencies
    2.4 dataset.

    Args:
        lazy (bool, optional): The reading strategy used when reading a
            dataset containing multiple documents. If this is true,
            ``iter()`` will return an object whose ``__iter__``
            method reloads the dataset each time it's called. Otherwise,
            ``iter()`` returns a list.
    """
    def __init__(self, lazy: bool = True):
        super().__init__(lazy)
        self.define_output_info()

    def define_output_info(self):
        self.output_info = {
            Document: [],
            Sentence: [],
            DependencyToken: ["universal_pos_tag", "features", "lemma",
                              "language_pos_tag", "misc"],
            # primary / enhanced dependencies
            UniversalDependency: ["type"]
        }

    def _cache_key_function(self, data_pack: DataPack) -> str:
        return data_pack.meta.doc_id

    def _collect(self, conll_directory) -> Iterator[Any]:
        """
        Iterator over conll files in the data_source
        :param conll_directory: directory to the conll files.
        :return: Iterator over files with conll path
        """
        token_comp_fields = ["id", "form", "lemma", "universal_pos_tag",
                             "language_pos_tag", "features", "head", "label",
                             "enhanced_dependency_relations", "misc"]

        token_multi_fields = ["features", "misc",
                              "enhanced_dependency_relations"]

        token_feature_fields = ["features", "misc"]

        token_entry_fields = ["lemma", "universal_pos_tag", "language_pos_tag",
                              "features", "misc"]

        file_paths = dataset_path_iterator(conll_directory, "conllu")
        for file_path in file_paths:
            with open(file_path, "r", encoding="utf8") as file:
                data_pack: DataPack
                doc_sent_begin: int
                doc_num_sent: int
                doc_offset: int
                doc_text: str
                doc_id: str
                sent_text: str
                sent_tokens: Dict[str, Tuple[Dict[str, Any], DependencyToken]] \
                    = {}

                lines = file.readlines()

                for i, line in enumerate(lines):
                    line = line.strip()
                    line_comps = line.split()

                    if line.startswith("# newdoc"):
                        data_pack = DataPack()
                        doc_sent_begin = 0
                        doc_num_sent = 0
                        doc_text = ''
                        doc_offset = 0
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

                        token: DependencyToken \
                            = DependencyToken(word_begin, word_end)
                        kwargs = {key: token_comps[key]
                                  for key in token_entry_fields}

                        # add token
                        token.set_fields(**kwargs)
                        data_pack.add_or_get_entry(token)

                        sent_tokens[str(token_comps["id"])] = (token_comps,
                                                               token)

                        sent_text += word + " "
                        doc_offset = word_end + 1

                    elif line == "":
                        # sentence ends
                        sent_text = sent_text.strip()
                        doc_text += ' ' + sent_text

                        # add dependencies for a sentence when all the tokens
                        # have been added
                        for token_id in sent_tokens:
                            token_comps, token = sent_tokens[token_id]

                            def add_dependency(head_id_, label_, dep_type,
                                               token_):
                                """
                                Adds dependency from :param head to token with
                                dependency label as :param label and type as :param
                                typ
                                """
                                if label_ == "root":
                                    token_.is_root = True
                                else:
                                    head = sent_tokens[head_id_][1]
                                    dependency = UniversalDependency(head,
                                                                     token_)
                                    dependency.dep_label = label_
                                    dependency.type = dep_type
                                    data_pack.add_or_get_entry(dependency)

                            # add primary dependency
                            add_dependency(token_comps["head"],
                                           token_comps["label"], "primary",
                                           token)

                            # add enhanced dependencies
                            for dep in token_comps[
                                            "enhanced_dependency_relations"]:
                                head_id, label = dep.split(":", 1)
                                add_dependency(head_id, label, "enhanced",
                                               token)

                        # add sentence
                        sent = Sentence(doc_sent_begin, doc_offset - 1)
                        data_pack.add_or_get_entry(sent)

                        doc_sent_begin = doc_offset
                        doc_num_sent += 1

                        if i == len(lines) - 1 or \
                                lines[i + 1].startswith("# newdoc"):
                            # doc ends
                            # add doc to data_pack
                            document = Document(0, len(doc_text))
                            data_pack.add_or_get_entry(document)
                            data_pack.meta.doc_id = doc_id
                            data_pack.set_text(doc_text.strip())

                            yield data_pack
                    else:
                        continue

    def parse_pack(self, data_pack) -> DataPack:
        return data_pack
