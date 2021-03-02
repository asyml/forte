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
import logging
import os
import pickle
import sys
from typing import Dict

from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.datasets.wikipedia.dbpedia.db_utils import load_redirects, \
    print_progress
from forte.datasets.wikipedia.dbpedia import (
    DBpediaWikiReader, WikiArticleWriter,
    WikiStructReader, WikiAnchorReader, WikiPropertyReader, WikiInfoBoxReader
)
from forte.data.readers.base_reader import PackReader
from forte.pipeline import Pipeline


def add_wiki_info(
        reader: PackReader, resources: Resources,
        input_path: str, input_pack_path: str,
        output_path: str, prompt_name: str, skip_existing=True):
    pl = Pipeline[DataPack](resources)

    if skip_existing and os.path.exists(output_path):
        print_progress(f'\n{output_path} exist, skipping {prompt_name}', '\n')
        return

    pl.set_reader(
        reader, config={
            'pack_index': os.path.join(input_pack_path, 'article.idx'),
            'pack_dir': input_pack_path,
        }
    )

    pl.add(
        WikiArticleWriter(), config={
            'output_dir': output_path,
            'zip_pack': True,
            'drop_record': True,
        },
    )

    print_progress(f'Start running the {prompt_name} pipeline.', '\n')
    pl.run(input_path)
    print_progress(f'Done collecting {prompt_name}.', '\n')


def read_wiki_text(
        nif_context: str,
        output_dir: str,
        resources: Resources,
        skip_existing: bool = False
):
    if skip_existing and os.path.exists(output_dir):
        print_progress(f'\n{output_dir} exist, skipping reading text', '\n')
        return

    pl = Pipeline[DataPack](resources)
    pl.set_reader(DBpediaWikiReader())
    pl.add(
        WikiArticleWriter(), config={
            'output_dir': output_dir,
            'zip_pack': True,
            'drop_record': True,
        },
    )
    print_progress('Start running wiki text pipeline.', '\n')
    pl.run(nif_context)
    print_progress('Done collecting wiki text.', '\n')


def main(nif_context: str, nif_page_structure: str, mapping_literals: str,
         mapping_objects: str, nif_text_links: str, redirects: str,
         info_boxs_properties: str, base_output_path: str):
    # The datasets are read in a few steps.
    # 0. Load redirects between wikipedia pages.
    print_progress('Loading redirects', '\n')
    redirect_pickle = os.path.join(base_output_path, 'redirects.pickle')

    redirect_map: Dict[str, str]
    if os.path.exists(redirect_pickle):
        redirect_map = pickle.load(open(redirect_pickle, 'rb'))
    else:
        redirect_map = load_redirects(redirects)
        with open(redirect_pickle, 'wb') as pickle_f:
            pickle.dump(redirect_map, pickle_f)

    resources: Resources = Resources()
    resources.update(redirects=redirect_map)
    print_progress("Done loading.", '\n')

    # 1. Read the wiki text.
    raw_pack_dir = os.path.join(base_output_path, 'nif_raw')
    read_wiki_text(nif_context, raw_pack_dir, resources, True)
    print_progress("Done reading wikipedia text.", '\n')

    # 2. Add the rest of wiki page structures:
    struct_dir = raw_pack_dir + '_struct'
    add_wiki_info(WikiStructReader(), resources, nif_page_structure,
                  raw_pack_dir, struct_dir, 'page_structures', True)
    print_progress("Done reading wikipedia structures.", '\n')

    link_dir = struct_dir + '_links'
    add_wiki_info(WikiAnchorReader(), resources, nif_text_links,
                  struct_dir, link_dir, 'anchor_links', True)
    print_progress("Done reading wikipedia anchors.", '\n')

    property_dir = link_dir + '_property'
    add_wiki_info(WikiPropertyReader(), resources, info_boxs_properties,
                  link_dir, property_dir, 'info_box_properties', True)
    print_progress("Done reading wikipedia info-boxes.", '\n')

    literal_dir = property_dir + '_literals'
    add_wiki_info(WikiInfoBoxReader(), resources, mapping_literals,
                  property_dir, literal_dir, 'literals', True)
    print_progress("Done reading wikipedia info-boxes literals.", '\n')

    mapping_dir = literal_dir + '_objects'
    add_wiki_info(WikiInfoBoxReader(), resources, mapping_objects,
                  literal_dir, mapping_dir, 'objects', True)
    print_progress("Done reading wikipedia info-boxes objects.", '\n')


def get_path(dataset: str):
    p = os.path.join(base_dir, dataset)
    if os.path.exists(p):
        return p
    else:
        raise FileNotFoundError(f'The dataset {dataset} is not found in '
                                f'base directory {base_dir}')


if __name__ == '__main__':
    base_dir = sys.argv[1]
    pack_output = os.path.join(base_dir, 'packs')

    if not os.path.exists(pack_output):
        os.makedirs(pack_output)

    logging.basicConfig(
        format='%(asctime)s - %(message)s', level=logging.INFO,
        filename=os.path.join(pack_output, 'dump.log')
    )

    main(
        get_path('nif_context_en.tql.bz2'),
        get_path('nif_page_structure_en.tql.bz2'),
        get_path('mappingbased_literals_en.tql.bz2'),
        get_path('mappingbased_objects_en.tql.bz2'),
        get_path('nif_text_links_en.tql.bz2'),
        get_path('redirects_en.tql.bz2'),
        get_path('infobox_properties_mapped_en.tql.bz2'),
        pack_output
    )
