"""
Tests for conllU reader
"""
import os
import unittest

from forte.data.ontology import universal_dependency_ontology
from forte.data.readers.conllu_ud_reader import ConllUReader


class ConllUReaderTest(unittest.TestCase):
    def setUp(self):
        """
        Reading the data into data_pack object to be used in the tests
        """
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.file_path = os.path.join(curr_dir, 'sample.conllu')
        self.multi_pack = ConllUReader().parse_pack(self.file_path).view()
        self.doc_ids = ["weblog-blogspot.com_nominations_20041117172713_ENG_"
                        "20041117_172713",
                        "weblog-blogspot.com_nominations_20041117172713_ENG_"
                        "20041117_172714"]

    def test_reader_text(self):
        doc_module = 'forte.data.ontology.base_ontology.Document'
        sent_module = 'forte.data.ontology.base_ontology.Sentence'

        expected_docs_text = [
            ["From the AP comes this story :",
             "President Bush on Tuesday nominated two individuals to "
             "replace retiring jurists on federal courts in the "
             "Washington area ."],
            ["Bush nominated Jennifer M. Anderson for a 15 - year "
             "term as associate judge of the Superior Court of the "
             "District of Columbia , replacing Steffen W. Graae ."]
        ]

        data_packs = self.multi_pack.packs
        self.assertEqual(len(data_packs), 2)

        for doc_index, doc_id in enumerate(self.doc_ids):
            data_pack = data_packs[doc_id]
            doc_entry = data_pack.get_entry_by_id(f"{doc_module}.{0}")

            expected_doc_text = expected_docs_text[doc_index]
            self.assertEqual(doc_entry.text, ' '.join(expected_doc_text))

            for sent_index, expected_sent_text in enumerate(expected_doc_text):
                sent_entry = data_pack.get_entry_by_id(
                    f"{sent_module}.{sent_index}")
                self.assertEqual(sent_entry.text, expected_sent_text)

    def test_reader_dependency_tree(self):
        data_pack = self.multi_pack.packs[self.doc_ids[1]]
        self.assertEqual(
            len(data_pack.get_entries_by_type(universal_dependency_ontology.Sentence)), 1)
        dependencies = data_pack.get_entries_by_type(universal_dependency_ontology.UniversalDependency)
        for link in dependencies:
            root_token = get_dependency_tree_root(link, data_pack)
            self.assertEqual(root_token.text, "nominated")


def get_dependency_tree_root(link, data_pack):
    """
    Returns the root token of the dependency tree in :param data_pack given an
    intermediate :param link
    """
    token = data_pack.get_entry_by_id(link.parent)
    if token.root:
        return token
    parent_link = list(data_pack.get_links_by_child(token))[0]
    return token if token.root else get_dependency_tree_root(parent_link,
                                                             data_pack)


if __name__ == "__main__":
    unittest.main()
