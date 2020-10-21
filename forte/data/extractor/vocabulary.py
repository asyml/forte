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
from collections import defaultdict


class Vocabulary:
    DEFAULT_PAD_ID = 0
    DEFAULT_PAD_ENTRY = "<PAD>"
    def __init__(self, method, pad_entry=None, pad_id=None):
        self.entry2id_dict = defaultdict()
        self.id2entry_dict = defaultdict()
        self.method = method
        if pad_entry is not None or \
            pad_id is not None:
            pad_entry = pad_entry if pad_entry is not None else self.DEFAULT_PAD_ENTRY
            pad_id = pad_id if pad_id is not None else self.DEFAULT_PAD_ID
            self.entry2id_dict[pad_entry] = pad_id
            self.id2entry_dict[pad_id] = pad_entry
        self.is_built = False

    def add_entry(self, entry):
        if entry not in self.entry2id_dict:
            idx = len(self.entry2id)
            self.entry2id_dict[entry] = idx
            self.id2entry_dict[idx] = entry

    def build(self):
        if not self.is_built:
            self.is_built = True

    def entry2id(self, entry: str):
        if not self.is_built:
            raise NotImplementedError("Vocab has not been built!")
        if self.method == "indexing":
            return self.entry2id_dict[entry]
        elif self.method == "one-hot":
            ans = [0 for _ in range(len(self.entry2id))]
            ans[self.entry2id_dict[entry]] = 1
            return ans

    def id2entry(self, idx: int):
        if not self.is_built:
            raise NotImplementedError("Vocab has not been built!")
        return self.id2entry_dict[idx]
