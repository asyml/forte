import os
import shutil
import tempfile
import unittest
from ddt import ddt, data

from forte.data.readers.plaintext_reader import PlainTextReader


@ddt
class PlainTextReaderTest(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.orig_text = get_sample_html_text()
        self.file_path = os.path.join(self.test_dir, 'test.html')
        self.mod_file_path = os.path.join(self.test_dir, 'mod_test.html')
        with open(self.file_path, 'w') as f:
            f.write(self.orig_text)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_reader_no_replace_test(self):
        # Read with no replacements
        pack = PlainTextReader().parse_pack(self.file_path)
        self.assertEqual(pack.text, self.orig_text)

    @data(
        # No replacement
        ([], '<title>The Original Title </title>'),
        # Insertion
        ([((11, 11), 'New ')], '<title>The New Original Title </title>'),
        # Single, sorted multiple and unsorted multiple replacements
        ([((11, 19), 'New')], '<title>The New Title </title>'),
        ([((0, 7), ''), ((26, 34), '')], 'The Original Title '),
        ([((26, 34), ''), ((0, 7), '')], 'The Original Title '),
    )
    def test_reader_replace_back_test(self, value):
        # Reading with replacements - replacing a span and changing it back
        span_ops, output = value
        reader = PlainTextReader()
        reader.text_replace_operation = lambda _: span_ops
        pack = reader.parse_pack(self.file_path)
        self.assertEqual(pack.text, output)

        with open(self.mod_file_path, 'w') as mod_file:
            mod_file.write(pack.text)
        reader.text_replace_operation = lambda _: pack.inverse_replace_operations
        inv_pack = reader.parse_pack(self.mod_file_path)
        self.assertEqual(self.orig_text, inv_pack.text)

    @data(
        ([((5, 8), ''), ((6, 10), '')], None),  # overlap
        ([((5, 8), ''), ((6, 1000), '')], None),  # outside limit
        ([((-1, 8), '')], None),  # does not support negative indexing
        ([((8, -1), '')], None),  # does not support negative indexing
        ([((2, 1), '')], None)  # start should be lesser than end
    )
    def test_reader_replace_error_test(self, value):
        # Read with errors in span replacements
        span_ops, output = value
        reader = PlainTextReader()
        reader.text_replace_operation = lambda _: span_ops
        try:

            reader.parse_pack(self.file_path)
        except ValueError:
            pass
        except Exception:
            self.fail('Unexpected exception raised:')
        else:
            self.fail('Expected Exception not raised')


def get_sample_html_text():
    return '<title>The Original Title </title>'


if __name__ == "__main__":
    unittest.main()
