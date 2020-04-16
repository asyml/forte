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
Unit tests for PlainTextReader.
"""
import os
import shutil
import tempfile
import unittest
from ddt import ddt, data

from forte.data.span import Span
from forte.data.readers import PlainTextReader
from forte.pack_manager import PackManager


@ddt
class PlainTextReaderTest(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.orig_text = "<title>The Original Title </title>"
        self.file_path = os.path.join(self.test_dir, 'test.html')
        self.mod_file_path = os.path.join(self.test_dir, 'mod_test.html')
        with open(self.file_path, 'w') as f:
            f.write(self.orig_text)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_reader_no_replace_test(self):
        # Read with no replacements
        reader = PlainTextReader()
        PackManager().set_input_source(reader.component_name)
        pack = list(reader.parse_pack(self.file_path))[0]
        self.assertEqual(pack.text, self.orig_text)

    @data(
        # No replacement
        ([], '<title>The Original Title </title>'),
        # Insertion
        ([(Span(11, 11), 'New ')], '<title>The New Original Title </title>'),
        # Single, sorted multiple and unsorted multiple replacements
        ([(Span(11, 19), 'New')], '<title>The New Title </title>'),
        ([(Span(0, 7), ''), (Span(26, 34), '')], 'The Original Title '),
        ([(Span(26, 34), ''), (Span(0, 7), '')], 'The Original Title '),
    )
    def test_reader_replace_back_test(self, value):
        # Reading with replacements - replacing a span and changing it back
        span_ops, output = value
        reader = PlainTextReader()
        reader.text_replace_operation = lambda _: span_ops
        pack = list(reader.parse_pack(self.file_path))[0]
        self.assertEqual(pack.text, output)

        orig_text_from_pack = pack.get_original_text()
        self.assertEqual(self.orig_text, orig_text_from_pack)

    @data(
        # before span starts
        (Span(1, 6), Span(1, 6), "relaxed"),
        (Span(1, 6), Span(1, 6), "strict"),
        # after span ends
        (Span(15, 22), Span(19, 21), "relaxed"),
        # span itself
        (Span(11, 14), Span(11, 19), "relaxed"),
        # complete string
        (Span(0, 40), Span(0, 34), "strict"),
        # cases ending to or starting from between the span
        (Span(11, 40), Span(11, 34), "relaxed"),
        (Span(13, 40), Span(11, 34), "relaxed"),
        (Span(14, 40), Span(19, 34), "relaxed"),
        (Span(13, 40), Span(11, 34), "backward"),
        (Span(13, 40), Span(19, 34), "forward"),
        (Span(0, 12), Span(0, 19), "relaxed"),
        (Span(0, 13), Span(0, 11), "backward"),
        (Span(0, 14), Span(0, 19), "forward"),
        # same begin and end
        (Span(38, 38), Span(32, 32), "relaxed"),
        (Span(38, 38), Span(32, 32), "strict"),
        (Span(38, 38), Span(32, 32), "backward"),
        (Span(38, 38), Span(32, 32), "forward")
    )
    def test_reader_original_span_test(self, value):
        span_ops, output = ([(Span(11, 19), 'New'),
                             (Span(19, 20), ' Shiny '),
                             (Span(25, 25), ' Ends')],
                            '<title>The New Shiny Title Ends </title>')
        input_span, expected_span, mode = value
        reader = PlainTextReader()
        PackManager().set_input_source(reader.component_name)
        reader.text_replace_operation = lambda _: span_ops
        pack = list(reader.parse_pack(self.file_path))[0]
        self.assertEqual(pack.text, output)

        output_span = pack.get_original_span(input_span, mode)
        self.assertEqual(output_span, expected_span,
                         f"Expected: ({expected_span.begin, expected_span.end}"
                         f"), Found: ({output_span.begin, output_span.end})"
                         f" when Input: ({input_span.begin, input_span.end})"
                         f" and Mode: {mode}")

    @data(
        ([(Span(5, 8), ''), (Span(6, 10), '')], None),  # overlap
        ([(Span(5, 8), ''), (Span(6, 1000), '')], None),  # outside limit
        ([(Span(-1, 8), '')], None),  # does not support negative indexing
        ([(Span(8, -1), '')], None),  # does not support negative indexing
        ([(Span(2, 1), '')], None)  # start should be lesser than end
    )
    def test_reader_replace_error_test(self, value):
        # Read with errors in span replacements
        span_ops, output = value
        reader = PlainTextReader()
        reader.text_replace_operation = lambda _: span_ops
        try:
            list(reader.parse_pack(self.file_path))[0]
        except ValueError:
            pass
        except Exception:
            self.fail('Unexpected exception raised:')
        else:
            self.fail('Expected Exception not raised')


if __name__ == "__main__":
    unittest.main()
