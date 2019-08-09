import unittest
import os
import shutil
import tempfile

from nlp.pipeline.data.ontology.top import Span
from nlp.pipeline.data.readers.html_reader import HTMLReader


class HTMLReaderTest(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_html_reader_replace_regex_test(self):
        # Test the replace function with only regex based replacements -
        # removing all tags
        text = get_sample_html_text()
        span_ops = [("</?[a-z]+>", '')]
        mod_text, inverse_ops = HTMLReader().replace(text, span_ops)
        self.assertEqual(mod_text, 'The Original Title HTML web page contents ')

    def test_html_reader_replace_back_regex_test(self):
        # Test the replace function with only regex based replacements -
        # removing all tags and changing it back to original
        text = get_sample_html_text()
        span_ops = [("</?[a-z]+>", '')]
        reader = HTMLReader()
        mod_text, inverse_ops = reader.replace(text, span_ops)
        orig_text, _ = reader.replace(mod_text, inverse_ops)
        self.assertEqual(orig_text, text)

    def test_html_replace_back_regex_span_test(self):
        # Test the replace function with regex and span based replacements -
        # removing all tags, replacing a span and changing it back to original
        text = get_sample_html_text()
        reader = HTMLReader()
        span_ops = [("</?[a-z]+>", ''), (Span(19, 31), 'The New')]
        mod_text, inverse_ops = reader.replace(text, span_ops)
        self.assertEqual(mod_text, 'The New Title HTML web page contents ')

        orig_text, _ = reader.replace(mod_text, inverse_ops)
        self.assertEqual(orig_text, text)

    def test_html_reader_replace_error_test(self):
        # Test the replace function with overlapping spans
        text = get_sample_html_text()
        reader = HTMLReader()
        # the span(5,8) overlaps with one of the tags
        span_ops = [("</?[a-z]+>", ''), (Span(5, 8), '')]
        try:
            reader.replace(text, span_ops)
        except ValueError:
            pass
        except Exception as exception:
            self.fail('Unexpected exception raised:', exception)
        else:
            self.fail('ExpectedException not raised')

    def test_html_reader_replace_test(self):
        # Test reading an html file with and without any replacement
        text = get_sample_html_text()
        file_path = 'test.html'
        with open(os.path.join(self.test_dir, file_path), 'w') as f:
            f.write(text)
        reader = HTMLReader()

        # no replacement
        pack = reader._read_document(file_path)
        self.assertEqual(pack.text, text)

        # regex replacement
        span_ops = [("</?[a-z]+>", '')]
        pack = reader._read_document(file_path, span_ops)
        expected_mod_text = 'The Original Title HTML web page contents '
        self.assertEqual(pack.text, expected_mod_text)


def get_sample_html_text():
    return '<html>' \
               '<head><title>The Original Title </title></head>' \
               '<body>HTML web page contents </body>' \
               '</html>'


if __name__ == "__main__":
    unittest.main()
