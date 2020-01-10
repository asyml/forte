import os
import unittest
from pathlib import Path
from ddt import ddt, data

from forte.pipeline import Pipeline
from forte.data.base import Span
from forte.data.readers import HTMLReader


@ddt
class HTMLReaderPipelineTest(unittest.TestCase):
    def setUp(self):
        self.reader = HTMLReader()

        self.pl1 = Pipeline()
        self._cache_directory = Path(os.path.join(os.getcwd(), "cache_data"))
        self.pl1.set_reader(HTMLReader())

    @data(
        ("<title>The Original Title </title>",
         "The Original Title "),
        ("<!DOCTYPE html><html><title>Page Title</title><body><p>This is a "
         "paragraph</p></body></html>",
         "Page TitleThis is a paragraph"),
        ('''<!DOCTYPE html>
    <html>
    <head>
    <title>Page Title</title>
    </head>
    <body>
    <h1>This is a Heading</h1>
    <p>This is a paragraph.</p>
    </body>
    </html>
    ''', '''
    \n    \n    Page Title\n    \n    \n    This is a Heading
    This is a paragraph.\n    \n    \n    '''),
        ('''<!DOCTYPE html>
    <h1 id="section1" class="bar">Section 1</h1>
    <p class="foo">foo bar\nbaz blah </p>
    <!-- cool beans! -->
    <hr/>
    <br>
    <p><em>The <strong>End!</strong></em></p>
    <p><em>error</p></em>weird < q <abc@example.com>
    ''', '''
    Section 1
    foo bar\nbaz blah \n    \n    \n    \n    The End!
    errorweird < q \n    '''),
        ('''<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN">
    <html<head>
    <title//
    <p ltr<span id=p>Text</span</p>
    </>''',
         '''\n    \n    Text
    ''')
    )
    def test_reader(self, value):
        html_input, expected_output = value
        doc_exists = False

        for pack in self.pl1.process_dataset(html_input):
            doc_exists = True
            self.assertEqual(expected_output, pack.text)
        self.assertTrue(doc_exists)

    @data(
        ('<title>The Original Title </title>'),
        ('''<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN">
    <html<head>
    <title//
    <p ltr<span id=p>Text</span</p>
    </>'''),
        ('''<!DOCTYPE html>
    <h1 id="section1" class="bar">Section 1</h1>
    <p class="foo">foo bar\nbaz blah </p>
    <!-- cool beans! -->
    <hr/>
    <br>
    <p><em>The <strong>End!</strong></em></p>
    <p><em>error</p></em>weird < q <abc@example.com>
    ''')
    )
    def test_reader_replace_back(self, value):
        input = value
        for pack in self.pl1.process_dataset(input):
            original_text = pack.get_original_text()
            self.assertEqual(original_text, input)

    @data(
        (Span(0, 3), Span(7, 10), '<title>The Original Title </title>',
         "strict"),
        (Span(18, 22), Span(101, 105), '''<!DOCTYPE html PUBLIC "-//W34.01//EN">
        <html<head>
        <title//
        <p ltr<span id=p>Text</span</p>
        </>''', "relaxed"),
        # # cases ending to or starting from between the span
        (Span(15, 30), Span(60, 95), '''<!DOCTYPE html>
        <h1 id="section1" class="bar">Section 1</h1>
        <p class="foo">foo bar\nbaz blah </p>
        <!-- cool beans! -->
        <hr/>
        <br>
        <p><em>The <strong>End!</strong></em></p>
        <p><em>error</p></em>weird < q <abc@example.com>''',
         "forward"),
        # before span starts
        (Span(0, 3), Span(0, 3), 'Some text<title>The Original Title </title>',
         "relaxed"),
        (Span(0, 3), Span(0, 3), 'Some text<title>The Original Title </title>',
         "strict"),
        # after span ends # There's an issue with this #TODO (assign) mansi
        # returns a span of (43, 35) which is wrong.
        # (Span(28, 28), Span(43, 43),
        #  'Some text<title>The Original Title </title>T',
        #  "strict"),
        # same begin and end
        (Span(14, 14), Span(21, 21),
         'Some text<title>The Original Title </title>',
         "strict"),
        (Span(14, 14), Span(21, 21),
         'Some text<title>The Original Title </title>',
         "relaxed"),
        (Span(14, 14), Span(21, 21),
         'Some text<title>The Original Title </title>',
         "backward"),
        (Span(14, 14), Span(21, 21),
         'Some text<title>The Original Title </title>',
         "forward")
    )
    def test_reader_original_span(self, value):
        new_span, expected_orig_span, html_input, mode = value
        for pack in self.pl1.process_dataset(html_input):
            # Retrieve original text
            original_text = pack.get_original_text()
            self.assertEqual(original_text, html_input)

            # Retrieve original span
            original_span = pack.get_original_span(new_span, mode)
            self.assertEqual(expected_orig_span, original_span)


if __name__ == '__main__':
    unittest.main()
