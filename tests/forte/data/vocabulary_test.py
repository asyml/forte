# Copyright 2020 The Forte Authors. All Rights Reserved.
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
import os
import unittest
import pickle as pkl
from itertools import product

from texar.torch.data import SpecialTokens

from forte.data import dataset_path_iterator
from forte.data.readers import plaintext_reader
from forte.data.vocabulary import Vocabulary, VocabFilter, FrequencyVocabFilter


class VocabularyTest(unittest.TestCase):
    def setUp(self):
        self.data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../../../', 'data_samples', 'random_texts'
        ))

    def argmax(self, one_hot):
        idx = -1
        for i, flag in enumerate(one_hot):
            if flag:
                self.assertTrue(idx == -1)
                idx = i
        return idx

    def test_vocabulary(self):
        methods = ["indexing", "one-hot"]
        flags = [True, False]
        for method, need_pad, use_unk in product(methods, flags, flags):
            vocab = Vocabulary[str](
                method=method, need_pad=need_pad, use_unk=use_unk)

            # Check vocabulary add_element, element2repr and id2element
            elements = ["EU", "rejects", "German", "call",
                        "to", "boycott", "British", "lamb", "."]
            for ele in elements:
                vocab.add_element(ele)
            save_len = len(vocab)
            for ele in elements:
                vocab.add_element(ele)
            self.assertEqual(save_len, len(vocab))

            representation = [vocab.element2repr(ele) for ele in elements]

            self.assertTrue(len(representation) > 0)

            if method == "indexing":
                self.assertTrue(isinstance(representation[0], int))
            else:
                self.assertTrue(isinstance(representation[0], list))

            recovered_elements = []
            for rep in representation:
                if method == "indexing":
                    idx = rep
                else:
                    idx = self.argmax(rep)
                recovered_elements.append(vocab.id2element(idx))

            self.assertListEqual(elements, recovered_elements)

            # Check __len__, items.
            self.assertEqual(len(set(elements)) + int(use_unk) +
                             int(need_pad),
                             len(vocab))
            saved_len = len(vocab)

            # Check has_element
            for ele in elements:
                self.assertTrue(vocab.has_element(ele))
            for ele in range(10):
                self.assertFalse(vocab.has_element(ele))

            # check PAD_ELEMENT
            if need_pad:
                if method == "indexing":
                    expected_pad_repr = 0
                else:
                    expected_pad_repr = [0] * (len(vocab) - 1)
                self.assertEqual(expected_pad_repr,
                                 vocab.element2repr(SpecialTokens.PAD))

            # Check UNK_ELEMENT
            if use_unk:
                if method == "indexing":
                    expected_unk_repr = 0 + int(need_pad)
                else:
                    expected_unk_repr = [0] * (len(vocab) - int(need_pad))
                    expected_unk_repr[0] = 1
                self.assertEqual(expected_unk_repr,
                                 vocab.element2repr(SpecialTokens.UNK))
                self.assertEqual(expected_unk_repr,
                                 vocab.element2repr("random_element"))
                self.assertEqual(saved_len, len(vocab))

            # Check state
            new_vocab = pkl.loads(pkl.dumps(vocab))
            self.assertEqual(vocab.method, new_vocab.method)
            self.assertEqual(vocab.need_pad, new_vocab.need_pad)
            self.assertEqual(vocab.use_unk, new_vocab.use_unk)
            self.assertEqual(vocab._element2id, new_vocab._element2id)
            self.assertEqual(vocab._id2element, new_vocab._id2element)
            self.assertEqual(vocab.next_id, new_vocab.next_id)

    def test_freq_filtering(self):
        base_vocab = Vocabulary()
        for p in dataset_path_iterator(self.data_path, '.txt'):
            with open(p) as f:
                for line in f:
                    for w in line.strip().split():
                        base_vocab.add_element(w)

        vocab_filter = FrequencyVocabFilter(
            base_vocab, min_frequency=2, max_frequency=4)

        filtered = base_vocab.filter(vocab_filter)

        for e, eid in base_vocab.items():
            base_count = base_vocab.get_count(e)
            if eid in base_vocab.special_ids:
                self.assertTrue(filtered.has_element(e))
                self.assertEqual(base_vocab.get_count(e), filtered.get_count(e))
            elif 2 <= base_count <= 4:
                print(e, base_vocab.get_count(e))
                self.assertTrue(filtered.has_element(e))
                self.assertEqual(base_count, filtered.get_count(e))
            else:
                self.assertFalse(filtered.has_element(e))


if __name__ == '__main__':
    unittest.main()
