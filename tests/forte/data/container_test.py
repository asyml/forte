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
Unit test for EntryContainer.
"""

import unittest

import os
import pickle
import tempfile

from forte.data.container import EntryContainer, E
from forte.data.ontology.core import Entry
from forte.data.span import Span


class DummyContainer(EntryContainer):
    def regret_creation(self, entry: E):
        pass

    def on_entry_creation(self, entry: Entry):
        pass

    def record_field(self, entry_id: int, field_name: str):
        pass

    def get_entry(self, tid: int):
        pass

    def get_span_text(self, span: Span):
        pass

    def validate(self, item):
        pass


class EntryContainerTest(unittest.TestCase):

    def test_pickle(self):
        container = DummyContainer()

        self.assertEqual(container.get_next_id(), 0)

        with tempfile.TemporaryDirectory() as tmpdirname:
            filename = os.path.join(tmpdirname, u"container.bin")
            with open(filename, "wb") as f:
                pickle.dump(container, f)
            with open(filename, "rb") as f:
                container_new = pickle.load(f)

        self.assertEqual(container_new.get_next_id(), 1)


if __name__ == '__main__':
    unittest.main()
