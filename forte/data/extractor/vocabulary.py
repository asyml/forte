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
from typing import Any

class Vocabulary:
    PAD_ENTRY = "<PAD>"
    UNK_ENTRY = "<UNK>"

    def __init__(self, method, use_pad, use_unk):
        self.entry2id_dict = defaultdict()
        self.id2entry_dict = defaultdict()

        if use_pad:
            self.add_entry(self.PAD_ENTRY)
        if use_unk:
            self.add_entry(self.UNK_ENTRY)
            self.id2entry_dict.default_factory = \
                lambda : self.UNK_ENTRY
            self.entry2id_dict.default_factory = \
                lambda : self.entry2id_dict[self.UNK_ENTRY]

        assert method in ("indexing", "one-hot")
        self.method = method
        self.use_pad = use_pad
        self.use_unk = use_unk

    def get_pad_id(self):
        if self.use_pad:
            return self.entry2id(self.PAD_ENTRY)
        return None

    def get_unk_id(self):
        if self.use_unk:
            return self.entry2id(self.UNK_ENTRY)
        return None

    def add_entry(self, entry: Any):
        if entry not in self.entry2id_dict:
            idx = len(self.entry2id_dict)
            self.entry2id_dict[entry] = idx
            self.id2entry_dict[idx] = entry

    def entry2id(self, entry: str):
        if self.method == "indexing":
            return self.entry2id_dict[entry]
        else:
            ans = [0 for _ in range(self.size())]
            ans[self.entry2id_dict[entry]] = 1
            return ans

    # TODO: Should we support one-hot to entry?
    def id2entry(self, idx: int):
        return self.id2entry_dict[idx]

    def size(self):
        return len(self.entry2id_dict)

    def contians(self, entry: Any):
        return entry in self.entry2id_dict

    def items(self):
        return self.entry2id_dict.items()
