import os
import unittest

from forte.datasets.wikipedia.dbpedia import db_utils
from forte.datasets.wikipedia.dbpedia.db_utils import (
    NIFParser, context_base, get_resource_fragment, get_resource_name,
    strip_url_params, ContextGroupedNIFReader
)


class DBUtilTest(unittest.TestCase):

    def setUp(self):
        self.data_dir: str = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../../../../data_samples/dbpedia'
        ))

    def test_redirect(self):
        redirect_path = os.path.join(self.data_dir, "redirects.tql")
        redirects = db_utils.load_redirects(redirect_path)
        self.assertEqual(redirects["AbeL"], "Cain_and_Abel")

    def test_nif_parser(self):
        p = os.path.join(self.data_dir, 'nif_page_structure.tql')

        parsed = []

        for statements in NIFParser(p):
            for statement in statements:
                s, v, o, c = statement
                parsed.append(
                    (context_base(c), get_resource_fragment(v),
                     get_resource_name(s), strip_url_params(s)))

        expected = [
            ('http://en.wikipedia.org/wiki/Animalia_(book)', 'type',
             'Animalia_(book)',
             'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)', 'notation',
                'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'beginIndex', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)', 'endIndex',
                'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'referenceContext', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'superString', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'hasSection', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'firstSection', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'lastSection', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)', 'type',
                'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'beginIndex', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)', 'endIndex',
                'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'referenceContext', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'superString', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'hasParagraph', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'lastParagraph', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)', 'type',
                'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)', 'notation',
                'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'beginIndex', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)', 'endIndex',
                'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'referenceContext', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'superString', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'hasSection', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'firstSection', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)', 'type',
                'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'referenceContext', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)',
                'beginIndex', 'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)'), (
                'http://en.wikipedia.org/wiki/Animalia_(book)', 'endIndex',
                'Animalia_(book)',
                'http://dbpedia.org/resource/Animalia_(book)')]
        self.assertEqual(parsed, expected)

    def test_grouped_nif_reader(self):
        p = os.path.join(self.data_dir, 'nif_context.tql')
        parsed = {}
        for context, statements in ContextGroupedNIFReader(p):
            for statement in statements:
                s, v, o = statement

                r = get_resource_fragment(v)
                n = get_resource_name(s)
                try:
                    parsed[get_resource_name(s)].append(r)
                except KeyError:
                    parsed[get_resource_name(s)] = [r]
        expected = {
            'Animalia_(book)': ['type', 'beginIndex', 'endIndex', 'sourceUrl',
                                'isString', 'predLang'],
            'List_of_Atlas_Shrugged_characters': ['type', 'beginIndex',
                                                  'endIndex', 'sourceUrl']
        }
        self.assertEqual(parsed, expected)
