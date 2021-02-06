# Copyright 2021 The Forte Authors. All Rights Reserved.
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
Unit test for dbpedia dataset.
"""
import glob
import os
import tempfile
from unittest import TestCase

from forte.common import Resources
from forte.data.data_pack import DataPack
from forte.datasets.wikipedia.dbpedia import DBpediaWikiReader, \
    WikiArticleWriter, WikiStructReader, WikiAnchorReader, WikiInfoBoxReader, \
    WikiPropertyReader
from forte.pipeline import Pipeline


def write_results(pl: Pipeline, output_path: str, input_data: str):
    pl.add(
        WikiArticleWriter(), config={
            'output_dir': output_path,
            'zip_pack': True,
            'drop_record': True,
        }
    )
    pl.run(input_data)


class TestDBpediaReaders(TestCase):
    """Test DBpedia Wikipedia readers."""

    def setUp(self):
        self.resources: Resources = Resources()
        self.resources.update(redirects={})

        self.data_dir: str = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../../../../data_samples/dbpedia'
        ))

        self.output_dir = tempfile.TemporaryDirectory()

        self.raw_output: str = os.path.join(self.output_dir.name, 'raw')

        pl = Pipeline[DataPack](self.resources)
        pl.set_reader(DBpediaWikiReader())
        pl.add(
            WikiArticleWriter(), config={
                'output_dir': self.raw_output,
                'zip_pack': True,
                'drop_record': True,
            }
        )
        pl.run(os.path.join(self.data_dir, 'nif_context.tql'))

    def num_packs_check(self, output: str, expected: int):
        num_packs_written: int = len(glob.glob(output + "/**/*.json.gz"))
        self.assertEqual(num_packs_written, expected)

    def num_indexed(self, output: str, expected: int):
        num_article_indices: int = sum(
            1 for _ in open(output + '/article.idx'))
        self.assertEqual(num_article_indices, expected)

    def test_wiki_text(self):
        self.num_packs_check(self.raw_output, 1)
        self.num_indexed(self.raw_output, 1)

    def test_struct(self):
        pl = Pipeline[DataPack](self.resources)
        pl.set_reader(
            WikiStructReader(), config={
                'pack_index': os.path.join(self.raw_output, 'article.idx'),
                'pack_dir': self.raw_output,
            }
        )

        output: str = os.path.join(self.output_dir.name, 'struct')
        write_results(
            pl, output,
            os.path.join(self.data_dir, 'nif_page_structure.tql')
        )
        self.num_packs_check(output, 1)
        self.num_indexed(output, 1)

    def test_anchor(self):
        pl = Pipeline[DataPack](self.resources)
        pl.set_reader(
            WikiAnchorReader(), config={
                'pack_index': os.path.join(self.raw_output, 'article.idx'),
                'pack_dir': self.raw_output,
            }
        )
        output: str = os.path.join(self.output_dir.name, 'anchor')
        write_results(
            pl, output,
            os.path.join(self.data_dir, 'text_links.tql')
        )

        self.num_packs_check(output, 1)
        self.num_indexed(output, 1)

    def test_property(self):
        pl = Pipeline[DataPack](self.resources)
        pl.set_reader(
            WikiPropertyReader(), config={
                'pack_index': os.path.join(self.raw_output, 'article.idx'),
                'pack_dir': self.raw_output,
            }
        )
        output: str = os.path.join(self.output_dir.name, 'property')
        write_results(
            pl, output,
            os.path.join(self.data_dir, 'info_box_property_mapped.tql')
        )

        self.num_packs_check(output, 1)
        self.num_indexed(output, 1)

    def test_infobox(self):
        pl = Pipeline[DataPack](self.resources)
        pl.set_reader(
            WikiInfoBoxReader(), config={
                'pack_index': os.path.join(
                    self.raw_output, 'article.idx'),
                'pack_dir': self.raw_output,
            }
        )
        output: str = os.path.join(self.output_dir.name, 'literals')
        write_results(
            pl, output,
            os.path.join(self.data_dir, 'literals.tql')
        )

        self.num_packs_check(output, 1)
        self.num_indexed(output, 1)

    def tearDown(self):
        self.output_dir.cleanup()
