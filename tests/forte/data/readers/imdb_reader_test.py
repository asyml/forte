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
Unit tests for IMDBReader.
"""
import os
import unittest
from typing import Iterator, Iterable, List
from forte.data.readers.imdb_reader import IMDBReader
from forte.data.data_pack import DataPack
from forte.pipeline import Pipeline
from ft.onto.base_ontology import Sentence, Token, Document


class IMDBReaderTest(unittest.TestCase):

    def setUp(self):
        # Define and config the pipeline.
        self.dataset_path: str = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            *([os.path.pardir] * 4),
            'data_samples/imdb'))

        self.pipeline: Pipeline = Pipeline[DataPack]()
        self.reader: IMDBReader = IMDBReader()
        self.pipeline.set_reader(self.reader)
        self.pipeline.initialize()

    def test_process_next(self):
        data_packs: Iterable[DataPack] = \
            self.pipeline.process_dataset(self.dataset_path)
        file_paths: Iterator[str] = \
            self.reader._collect(self.dataset_path)

        count_packs: int = 0

        # Each .imdb file is corresponding to an Iterable Obj
        for pack, file_path in zip(data_packs, file_paths):

            count_packs += 1
            expected_doc: str = ""

            # Read all lines in .imdb file
            with open(file_path, "r", encoding="utf8", errors='ignore') as file:
                expected_doc = file.read()

            # Test document.
            actual_docs: List[Document] = list(pack.get(Document))
            self.assertEqual(len(actual_docs), 1)

            lines: List[str] = expected_doc.split('\n')
            comment_lines = []
            sentiment_labels = []
            wordlist = []
            for line in lines:
                tempwordlist = []
                #for empty or invalid line
                if len(line) < 5:
                    continue
                comment = line.split("\",")[0].strip("\"")
                sentiment_label = line.split("\",")[1]
                comment_lines.append(comment)
                sentiment_labels.append(sentiment_label)

                tempwordlist = comment.split(" ")
                for w in tempwordlist:
                    wordlist.append(w)

            actual_sentences: Iterator[Sentence] = pack.get(Sentence)
            actual_word: Iterator[Token] = pack.get(Token)
            # Force sorting as Link entries have no order when retrieving from
            # data pack.
            for line, label, actual_sentence in \
                    zip(comment_lines, sentiment_labels, actual_sentences):
                line = line.strip()
                label = label
                comment = actual_sentence.text
                # Test comment.
                self.assertEqual(comment,line)
                self.assertEqual(actual_sentence.speaker, label)
                #self.assertEqual(actual_sentence.__getattribute__("sentim"), label)


            for word_read, word_in_pack in zip(wordlist,actual_word):
                new_word_read = word_read
                lastch = word_read[len(word_read) - 1]
                if lastch == "," or lastch=='.':
                    new_word_read = word_read[:len(word_read) - 1]
                self.assertEqual(new_word_read, word_in_pack.text)

        self.assertEqual(count_packs, 1)


if __name__ == '__main__':
    unittest.main()