import unittest
import os
import shutil
import tempfile

from forte.data.readers.plaintext_reader import PlainTextReader


class HTMLReaderTest(unittest.TestCase):
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

    def test_html_reader_replace_regex_test(self):
        # Test the replace function with only regex based replacements -
        # removing all tags
        span_ops = [("</?[a-z]+>", '')]
        pack = PlainTextReader().read(self.file_path, span_ops)
        self.assertEqual(pack.text, 'The Original Title HTML web page contents ')

    def test_html_reader_replace_back_regex_test(self):
        # Test the replace function with only regex based replacements -
        # removing all tags and changing it back to original
        reader = PlainTextReader()
        span_ops = [("</?[a-z]+>", '')]
        pack = reader.read(self.file_path, span_ops)
        with open(self.mod_file_path, 'w') as mod_file:
            mod_file.write(pack.text)
        inv_pack = reader.read(self.mod_file_path,
                               pack.inverse_replace_operations)
        self.assertEqual(self.orig_text, inv_pack.text)

    def test_html_replace_back_regex_span_test(self):
        # Test the replace function with regex and span based replacements -
        # removing all tags, replacing a span and changing it back to original
        reader = PlainTextReader()
        span_ops = [("</?[a-z]+>", ''), ((19, 31), 'The New')]
        pack = reader.read(self.file_path, span_ops)
        self.assertEqual(pack.text, 'The New Title HTML web page contents ')
        with open(self.mod_file_path, 'w') as mod_file:
            mod_file.write(pack.text)
        inv_pack = reader.read(self.mod_file_path,
                               pack.inverse_replace_operations)
        self.assertEqual(self.orig_text, inv_pack.text)

    def test_html_reader_replace_error_test(self):
        # Test the replace function with overlapping spans
        reader = PlainTextReader()
        # the span(5,8) overlaps with one of the tags
        span_ops = [("</?[a-z]+>", ''), ((5, 8), '')]
        try:
            reader.read(self.file_path, span_ops)
        except ValueError:
            pass
        except Exception as exception:
            self.fail('Unexpected exception raised:', exception)
        else:
            self.fail('ExpectedException not raised')


def get_sample_html_text():
    return '<html>' \
               '<head><title>The Original Title </title></head>' \
               '<body>HTML web page contents </body>' \
               '</html>'


if __name__ == "__main__":
    unittest.main()
