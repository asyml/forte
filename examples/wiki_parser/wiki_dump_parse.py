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
from typing import Dict, Optional

from forte.common.resources import Resources
from forte.data.data_pack import DataPack
from forte.datasets.wikipedia.dbpedia.db_utils import (
    load_redirects,
    print_progress,
)
from forte.datasets.wikipedia.dbpedia import (
    DBpediaWikiReader,
    WikiArticleWriter,
    WikiStructReader,
    WikiAnchorReader,
    WikiPropertyReader,
    WikiInfoBoxReader,
)
from forte.data.base_reader import PackReader
from forte.datasets.wikipedia.dbpedia.dbpedia_datasets import (
    WikiCategoryReader,
    WikiPackReader,
)
from forte.pipeline import Pipeline


def add_wiki_info(
    reader: WikiPackReader,
    resources: Resources,
    wiki_info_data_path: str,
    input_pack_path: str,
    output_path: str,
    prompt_name: str,
    use_input_index=False,
    skip_existing=True,
    resume_from_last=False,
    input_index_file_path: Optional[str] = "article.idx",
    output_index_file_name: Optional[str] = "article.idx",
):
    """
    Add wiki resource into the data pack.

    Args:
        reader: The info reader that loads the data pack.
        resources: The resources object that should contain the redirects.
        wiki_info_data_path: The path containing the wiki data.
        input_pack_path: The initial data pack path.
        output_path: The resulting output path.
        prompt_name: a name to show during processing.
        use_input_index: whether to use the input index to determine the
          output path.
        skip_existing: whether to skip this function if the folder exists.
        resume_from_last: whether to resume from last end point, at most one
          can be true between this and `skip_existing`
        input_index_file_path: the full file path to the input index.
        output_index_file_name: the file path to write the output index,
            this is relative to `output_path`.

    Returns:

    """
    pl = Pipeline[DataPack](resources)

    if resume_from_last and skip_existing:
        raise ValueError(
            "resume_from_last and skip_existing cannot both be " "true."
        )

    out_index_path = os.path.join(output_path, output_index_file_name)
    if skip_existing and os.path.exists(out_index_path):
        print_progress(
            f"\n{out_index_path} exist, skipping {prompt_name}", "\n"
        )
        return

    if resume_from_last:
        if not os.path.exists(out_index_path):
            raise ValueError(f"Configured to do resume but path "
                             f"{out_index_path} does not exists.")

        print_progress(
            f"\nWill resume from last from {out_index_path}", "\n"
        )
        pl.set_reader(
            reader,
            config={
                "pack_index": input_index_file_path,
                "pack_dir": input_pack_path,
                "resume_index": out_index_path,
                "zip_pack": True,
            },
        )
    else:
        pl.set_reader(
            reader,
            config={
                "pack_index": input_index_file_path,
                "pack_dir": input_pack_path,
                "zip_pack": True,
            },
        )

    pl.add(
        WikiArticleWriter(),
        config={
            "output_dir": output_path,
            "zip_pack": True,
            "drop_record": True,
            "use_input_index": use_input_index,
            "input_index_file": input_index_file_path,
            "output_index_file": output_index_file_name,
            "append_to_index": resume_from_last,
        },
    )

    print_progress(f"Start running the {prompt_name} pipeline.", "\n")
    pl.run(wiki_info_data_path)
    print_progress(f"Done collecting {prompt_name}.", "\n")


def read_wiki_text(
    nif_context: str,
    output_dir: str,
    resources: Resources,
    skip_existing: bool = False,
):
    if skip_existing and os.path.exists(output_dir):
        print_progress(f"\n{output_dir} exist, skipping reading text", "\n")
        return

    pl = Pipeline[DataPack](resources)
    pl.set_reader(DBpediaWikiReader())
    pl.add(
        WikiArticleWriter(),
        config={
            "output_dir": output_dir,
            "zip_pack": True,
            "drop_record": True,
        },
    )
    print_progress("Start running wiki text pipeline.", "\n")
    pl.run(nif_context)
    print_progress("Done collecting wiki text.", "\n")


def cache_redirects(
    base_output_path: str, redirect_path: str
) -> Dict[str, str]:
    redirect_pickle = os.path.join(base_output_path, "redirects.pickle")

    redirect_map: Dict[str, str]
    if os.path.exists(redirect_pickle):
        redirect_map = pickle.load(open(redirect_pickle, "rb"))
    else:
        redirect_map = load_redirects(redirect_path)
        with open(redirect_pickle, "wb") as pickle_f:
            pickle.dump(redirect_map, pickle_f)
    return redirect_map


def main(
    nif_context: str,
    nif_page_structure: str,
    mapping_literals: str,
    mapping_objects: str,
    nif_text_links: str,
    redirects: str,
    info_boxs_properties: str,
    categories: str,
    base_output_path: str,
    resume_existing: bool,
):
    # Whether to skip the whole step.
    if resume_existing:
        skip_existing = False
    else:
        skip_existing = True

    # The datasets are read in a few steps.
    # 0. Load redirects between wikipedia pages.
    print_progress("Loading redirects", "\n")

    redirect_map: Dict[str, str] = cache_redirects(base_output_path, redirects)

    resources: Resources = Resources()
    resources.update(redirects=redirect_map)
    print_progress("Done loading.", "\n")

    # 1. Read the wiki text.
    raw_pack_dir = os.path.join(base_output_path, "nif_raw")
    read_wiki_text(nif_context, raw_pack_dir, resources, True)
    print_progress("Done reading wikipedia text.", "\n")

    # Use the same index structure for all writers.
    main_index = os.path.join(raw_pack_dir, "article.idx")

    # 2. Add wiki page structures, create a new directory for it.
    struct_dir = raw_pack_dir + "_struct"
    add_wiki_info(
        WikiStructReader(),
        resources,
        nif_page_structure,
        raw_pack_dir,
        struct_dir,
        "page_structures",
        use_input_index=True,
        skip_existing=skip_existing,
        resume_from_last=resume_existing,
        input_index_file_path=main_index,
    )
    print_progress("Done reading wikipedia structures.", "\n")

    # 3. Add wiki links, create a new directory for it.
    link_dir = struct_dir + "_links"
    add_wiki_info(
        WikiAnchorReader(),
        resources,
        nif_text_links,
        struct_dir,
        link_dir,
        "anchor_links",
        use_input_index=True,
        skip_existing=True,
        resume_from_last=resume_existing,
        input_index_file_path=main_index,
    )
    print_progress("Done reading wikipedia anchors.", "\n")

    # 4 The following steps add info boxes:
    # 4.1 Add un-mapped infobox, we directly write to the previous directory
    property_dir = link_dir
    add_wiki_info(
        WikiPropertyReader(),
        resources,
        info_boxs_properties,
        link_dir,
        property_dir,
        "info_box_properties",
        use_input_index=True,
        skip_existing=True,
        resume_from_last=resume_existing,
        output_index_file_name="properties.idx",
        input_index_file_path=main_index,
    )
    print_progress("Done reading wikipedia info-boxes properties.", "\n")

    # 4.1 Add mapped literal, we directly write to the previous directory.
    literal_dir = property_dir
    add_wiki_info(
        WikiInfoBoxReader(),
        resources,
        mapping_literals,
        property_dir,
        literal_dir,
        "literals",
        use_input_index=True,
        skip_existing=True,
        resume_from_last=resume_existing,
        output_index_file_name="literals.idx",
        input_index_file_path=main_index,
    )
    print_progress("Done reading wikipedia info-boxes literals.", "\n")

    # 4.1 Add mapped object, we directly write to the previous directory.
    mapping_dir = literal_dir
    add_wiki_info(
        WikiInfoBoxReader(),
        resources,
        mapping_objects,
        literal_dir,
        mapping_dir,
        "objects",
        use_input_index=True,
        skip_existing=True,
        resume_from_last=resume_existing,
        output_index_file_name="objects.idx",
        input_index_file_path=main_index,
    )
    print_progress("Done reading wikipedia info-boxes objects.", "\n")

    # 4.2 Add category, directly write to previous directory.
    category_dir = mapping_dir
    add_wiki_info(
        WikiCategoryReader(),
        resources,
        categories,
        mapping_dir,
        category_dir,
        "categories",
        use_input_index=True,
        skip_existing=True,
        resume_from_last=resume_existing,
        output_index_file_name="categories.idx",
        input_index_file_path=main_index,
    )


def get_path(dataset: str):
    p = os.path.join(base_dir, dataset)
    if os.path.exists(p):
        return p
    else:
        raise FileNotFoundError(
            f"The dataset {dataset} is not found in "
            f"base directory {base_dir}"
        )


if __name__ == "__main__":
    base_dir = sys.argv[1]
    pack_output = sys.argv[2]

    try_samples = False
    if len(sys.argv) > 3:
        with_samples = sys.argv[3]
        try_samples = with_samples.upper().startswith("TRUE")

    will_resume = False
    if len(sys.argv) > 4:
        resume = sys.argv[4]
        will_resume = resume.upper().startswith("TRUE")

    if not os.path.exists(pack_output):
        os.makedirs(pack_output)

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        filename=os.path.join(pack_output, "dump.log"),
    )

    if try_samples:
        main(
            get_path("nif_context.tql"),
            get_path("nif_page_structure.tql"),
            get_path("literals.tql"),
            get_path("mappingbased_objects_en.tql"),
            get_path("text_links.tql"),
            get_path("redirects.tql"),
            get_path("infobox_properties_mapped_en.tql"),
            get_path("article_categories_en.tql"),
            pack_output,
            will_resume,
        )
    else:
        main(
            get_path("nif_context_en.tql.bz2"),
            get_path("nif_page_structure_en.tql.bz2"),
            get_path("mappingbased_literals_en.tql.bz2"),
            get_path("mappingbased_objects_en.tql.bz2"),
            get_path("nif_text_links_en.tql.bz2"),
            get_path("redirects_en.tql.bz2"),
            get_path("infobox_properties_mapped_en.tql.bz2"),
            get_path("article_categories_en.tql.bz2"),
            pack_output,
            will_resume,
        )
