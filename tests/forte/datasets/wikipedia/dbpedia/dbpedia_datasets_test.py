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
    WikiArticleWriter, WikiStructReader
from forte.pipeline import Pipeline


class TestDBpediaReaders(TestCase):
    """Test DBpedia Wikipedia readers."""

    def setUp(self):
        resources: Resources = Resources()
        resources.update(redirects={})

        self.data_dir: str = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            '../../../../../data_samples/dbpedia'
        ))

        self.output_dir = tempfile.TemporaryDirectory()

        self.pl = Pipeline[DataPack](resources)

    def test_wiki_text(self):
        output: str = os.path.join(self.output_dir.name, 'context')
        self.pl.set_reader(DBpediaWikiReader())
        self.pl.add(
            WikiArticleWriter(), config={
                'output_dir': output,
                'zip_pack': True,
                'drop_record': True,
            }
        )
        self.pl.initialize()

        num_articles: int = 0
        for _ in self.pl.process_dataset(
                os.path.join(self.data_dir, 'nif_context_preview.tql')):
            num_articles += 1
        self.pl.finish()

        num_packs_written: int = len(glob.glob(output + "/**/*.json.gz"))
        num_article_indices: int = sum(
            1 for _ in open(output + '/article.idx'))

        self.assertEqual(num_articles, 1)
        self.assertEqual(num_packs_written, 1)
        self.assertEqual(num_article_indices, 1)

    # def test_wiki_struct(self):
    #     in_dir: str = os.path.join(self.output_dir.name, 'context')
    #
    #     self.pl.set_reader(
    #         WikiStructReader(), config={
    #             'pack_index': os.path.join(in_dir, 'article.idx'),
    #             'pack_dir': in_dir,
    #         }
    #     )

    def tearDown(self):
        self.output_dir.cleanup()
