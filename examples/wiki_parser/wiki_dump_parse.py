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
This creates a pipeline to parse the Wikipedia dump and save the results
as MultiPacks onto disk.
"""
import csv
import logging
import os
import pickle
import sys
from typing import TextIO, Any, Dict

from texar.torch.hyperparams import HParams

from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.data.datasets.wikipedia.db_utils import load_redirects
from forte.data.datasets.wikipedia.dbpedia_based_reader import DBpediaWikiReader
from forte.data.datasets.wikipedia.dbpedia_infobox_reader import \
    DBpediaInfoBoxReader
from forte.pipeline import Pipeline
from forte.processors.base.writers import JsonPackWriter
from ft.onto.wikipedia import WikiPage

__all__ = [
    'WikiArticleWriter',
]


class WikiArticleWriter(JsonPackWriter[DataPack]):
    article_index: TextIO

    # It is difficult to get the type of the csv writer
    # https://stackoverflow.com/questions
    # /51264355/how-to-type-annotate-object-returned-by-csv-writer
    csv_writer: Any

    def __init__(self):
        super().__init__()
        self.article_count: int = 0

    def initialize(self, resources: Resources, configs: HParams):
        super(WikiArticleWriter, self).initialize(resources, configs)
        self.article_count = 0
        self.article_index = open(
            os.path.join(self.root_output_dir, 'article.idx'), 'w')
        self.csv_writer = csv.writer(self.article_index, delimiter='\t')

    def sub_output_path(self, pack: DataPack) -> str:
        sub_dir = str(int(self.article_count / 2000)).zfill(5)
        pid = pack.get_single(WikiPage).page_id
        doc_name = f'doc_{self.article_count}' if pid is None else pid

        return os.path.join(sub_dir, doc_name + '.json')

    def _process(self, input_pack: DataPack):
        """
        Write an index from the document id to the relative storage of this
        DataPack. This can be used as a simple index to retrieve the relevant
        file, which can enable faster lookup in use cases like following the
        Wikipedia links.

        Args:
            input_pack: The DataPack that contains the Wikipedia information.

        Returns:
        """
        super(WikiArticleWriter, self)._process(input_pack)

        out_path = self.sub_output_path(input_pack)
        if self.zip_pack:
            out_path = out_path + '.gz'

        # Write the index
        self.csv_writer.writerow([input_pack.meta.doc_id, out_path])
        self.article_count += 1

        if self.article_count % 1000 == 0:
            logging.info(
                "Written %s to %s",
                self.article_count, self.root_output_dir
            )

    def finish(self, resource: Resources):
        # pylint: disable=unused-argument
        self.article_index.close()


def main(nif_context: str, nif_page_structure: str, mapping_literals: str,
         mapping_objects: str, nif_text_links: str, redirects: str,
         info_boxs: str, output_path: str):
    # Load redirects.
    logging.info("Loading redirects")
    redirect_pickle = os.path.join(output_path, 'redirects.pickle')

    redirect_map: Dict[str, str]
    if os.path.exists(redirect_pickle):
        redirect_map = pickle.load(open(redirect_pickle, 'rb'))
    else:
        redirect_map = load_redirects(redirects)
        with open(redirect_pickle, 'wb') as pickle_f:
            pickle.dump(redirect_map, pickle_f)
    logging.info("Done loading.")

    # The datasets are read in two steps.
    raw_pack_dir = os.path.join(output_path, 'nif_raw')

    # First, we create the NIF reader that read the NIF in order.
    nif_pl = Pipeline()
    nif_pl.resource.update(redirects=redirect_map)

    nif_pl.set_reader(DBpediaWikiReader(), config=HParams(
        {
            'redirect_path': redirects,
            'nif_page_structure': nif_page_structure,
            'nif_text_links': nif_text_links,
        },
        DBpediaWikiReader.default_configs()
    ))

    nif_pl.add_processor(WikiArticleWriter(), config=HParams(
        {
            'output_dir': raw_pack_dir,
            'zip_pack': True,
        },
        WikiArticleWriter.default_configs()
    ))

    nif_pl.initialize()
    logging.info('Start running the DBpedia text pipeline.')
    nif_pl.run(nif_context)

    # Second, we add info boxes to the packs with NIF.
    ib_pl = Pipeline()
    ib_pl.resource.update(redirects=redirect_map)
    ib_pl.set_reader(DBpediaInfoBoxReader(), config=HParams(
        {
            'pack_index': os.path.join(raw_pack_dir, 'article.idx'),
            'pack_dir': raw_pack_dir,
            'mapping_literals': mapping_literals,
            'mapping_objects': mapping_objects,
            'reading_log': os.path.join(output_path, 'infobox.log')
        },
        DBpediaInfoBoxReader.default_configs()
    ))

    ib_pl.add_processor(WikiArticleWriter(), config=HParams(
        {
            'output_dir': os.path.join(output_path, 'nif_info_box'),
            'zip_pack': True,
        },
        WikiArticleWriter.default_configs()
    ))

    # Now we run the info box pipeline.
    ib_pl.initialize()
    ib_pl.run(info_boxs)


def get_data(dataset: str):
    p = os.path.join(base_dir, dataset)
    if os.path.exists(p):
        return p
    else:
        raise FileNotFoundError(f'The dataset {dataset} is not found in '
                                f'base directory {base_dir}')


if __name__ == '__main__':
    base_dir = sys.argv[1]

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    main(
        get_data('nif_context_en.tql.bz2'),
        get_data('nif_page_structure_en.tql.bz2'),
        get_data('mappingbased_literals_en.tql.bz2'),
        get_data('mappingbased_objects_en.tql.bz2'),
        get_data('nif_text_links_en.tql.bz2'),
        get_data('redirects_en.tql.bz2'),
        get_data('infobox_properties_mapped_en.tql.bz2'),
        os.path.join(base_dir, 'packs'),
    )
