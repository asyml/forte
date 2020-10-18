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
    def __init__(self, method, use_unk = False, unk_entry = "<UNK>"):
        self.is_built = False
        self.use_unk = use_unk
        self.method = method

        if use_unk:
            self.unk_id = 0
            self.unk_entry = unk_entry
            self.entry2id = defaultdict(lambda : self.unk_id)
            self.id2entry = defaultdict(lambda : self.unk_entry)
            self.entry2id[self.unk_entry] = self.unk_id
            self.id2entry[0] = self.unk_entry
        else:
            self.entry2id = defaultdict()
            self.id2entry = defaultdict()

    def add_entry(self, entry):
        if entry not in self.entry2id:
            idx = len(self.entry2id)
            self.entry2id[entry] = idx
            self.id2entry[idx] = entry

    def build(self):
        if not self.is_built:
            self.is_built = True

    def to_id(self, entry: str):
        if not self.is_built:
            raise NotImplementedError("Vocab has not been built!")
        if self.method == "indexing":
            return self.entry2id(entry)
        elif self.method == "one-hot":
            ans = [0 for _ in range(len(self.entry2id))]
            ans[self.entry2id(entry)] = 1
            return ans

    def from_id(self, idx: int):
        if not self.is_built:
            raise NotImplementedError("Vocab has not been built!")
        return self.from_id(idx)
