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
In this module, we provide a few readers to help reading DBpedia processed
 Wikipedia dumps. To use the readers here, the DBpedia full text datasets
 are needed: https://wiki.dbpedia.org/downloads-2016-10#p10608-2
"""
import csv
import logging
import os
from collections import defaultdict
from typing import (
    Iterator,
    Dict,
    List,
    Tuple,
    TextIO,
    Any,
    DefaultDict,
    Optional,
)

import rdflib
from smart_open import open

from forte.common import Resources
from forte.common.configuration import Config
from forte.common.exception import ResourceError
from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from forte.datasets.wikipedia.dbpedia.db_utils import (
    NIFParser,
    get_resource_attribute,
    get_resource_name,
    get_resource_fragment,
    print_progress,
    ContextGroupedNIFReader,
    state_type,
)
from forte.processors.base import JsonPackWriter
from ft.onto.wikipedia import (
    WikiPage,
    WikiSection,
    WikiParagraph,
    WikiTitle,
    WikiAnchor,
    WikiInfoBoxProperty,
    WikiInfoBoxMapped,
)

__all__ = [
    "DBpediaWikiReader",
    "WikiPackReader",
    "WikiArticleWriter",
    "WikiAnchorReader",
    "WikiStructReader",
    "WikiPropertyReader",
    "WikiInfoBoxReader",
]


class DBpediaWikiReader(PackReader):
    """
    This reader reads in the Wikipedia full text articles from a DBpedia
    full text dump, which is the `NIF Context` dataset from here:
    https://wiki.dbpedia.org/downloads-2016-10#p10608-2 .
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.__redirects: Dict[str, str] = {}

    def initialize(self, resources: Resources, config: Config):
        super().initialize(resources, config)
        if self.resources.contains("redirects"):
            self.__redirects = self.resources.get("redirects")
            logging.info("%d redirects loaded.", len(self.__redirects))
        else:
            raise ResourceError("Redirects not provided from resources.")

    def _collect(  # type: ignore
        self, nif_context: str
    ) -> Iterator[Dict[str, str]]:
        str_data: Dict[str, str] = {}

        for context_statements in NIFParser(nif_context):
            for s, v, o, c in context_statements:
                nif_type = get_resource_attribute(s, "nif")
                print_progress(f"Collecting DBpedia resource: [{c.identifier}]")

                fragment = get_resource_fragment(v)
                if (
                    nif_type
                    and nif_type == "context"
                    and fragment is not None
                    and fragment == "isString"
                ):
                    str_data["text"] = o.toPython()
                    doc_name: Optional[str] = get_resource_name(s)
                    old_id: Optional[str] = get_resource_attribute(
                        c.identifier, "oldid"
                    )
                    if doc_name is not None and old_id is not None:
                        str_data["doc_name"] = doc_name
                        str_data["oldid"] = old_id
                        yield str_data

    def _parse_pack(self, doc_data: Dict[str, str]) -> Iterator[DataPack]:
        pack = DataPack()
        doc_name: str = doc_data["doc_name"]
        if doc_name in self.__redirects:
            doc_name = self.__redirects[doc_name]

        full_text: str = doc_data["text"]

        pack.set_text(full_text)
        page = WikiPage(pack, 0, len(full_text))
        page.page_id = doc_data["oldid"]
        page.page_name = doc_name
        pack.pack_name = doc_name
        yield pack


def read_index(pack_index_path: str) -> Dict[str, str]:
    """
    Reads an index from the page name to the path of the stored pack.

    Args:
        pack_index_path: The path of this index file. The file should be a tab
        separated file.

    Returns:
        A dictionary that maps from the page name to the full path.
    """
    page_idx: Dict[str, str] = {}

    logging.info("Reading pack index from %s", pack_index_path)

    with open(pack_index_path) as idx:
        for page_name, page_path in csv.reader(idx, delimiter="\t"):
            page_idx[page_name] = page_path
    return page_idx


class WikiPackReader(PackReader):
    """
    This reader reads information from an NIF graph, and find out the
    corresponding data pack stored on disk. The output from this reader
    are these data packs plus the additional NIF information.

    The function `add_wiki_info` is to be implemented to handle how the
    NIF statements are added to the data pack.
    """

    def __init__(self):
        super().__init__()
        self._pack_index: Dict[str, str] = {}
        self._pack_dir: str = ""
        self._redirects: Dict[str, str] = {}

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        # A mapping from the name of the page to the path on th disk.
        self._pack_index = read_index(configs.pack_index)
        self._pack_dir = configs.pack_dir

        if self.resources.contains("redirects"):
            self._redirects = self.resources.get("redirects")
            logging.info("%d redirects loaded.", len(self._redirects))
        else:
            raise ResourceError("Redirects not provided from resources.")

    def add_wiki_info(self, pack: DataPack, statements: List[state_type]):
        raise NotImplementedError

    def _collect(  # type: ignore
        self, nif_path: str
    ) -> Iterator[Tuple[str, Dict[str, List[state_type]]]]:
        for _, statements in ContextGroupedNIFReader(nif_path):
            name = get_resource_name(statements[0][0])
            if name is not None:
                yield name, statements

    def _parse_pack(
        self, collection: Tuple[str, List[state_type]]
    ) -> Iterator[DataPack]:
        resource_name, statements = collection
        if resource_name in self._redirects:
            resource_name = self._redirects[resource_name]

        if resource_name in self._pack_index:
            print_progress(
                f"Handling resource [{resource_name}] in {self.component_name}"
            )
            pack_path = os.path.join(
                self._pack_dir, self._pack_index[resource_name]
            )

            # `smart_open` can handle the `gz` files.
            if os.path.exists(pack_path):
                with open(pack_path) as pack_file:
                    pack: DataPack = DataPack.deserialize(pack_file.read())
                    self.add_wiki_info(pack, statements)
                    yield pack
        else:
            logging.info("Resource %s pack not found.", resource_name)

    @classmethod
    def default_configs(cls):
        """
        This defines a basic config structure for the reader.

        Here:
          - pack_dir: the directory that contains all the serialized packs.
          - pack_index: the file name under the pack directory that points to
            the index from the name to the actual pack path.

        :return:
        """
        config = super().default_configs()
        config.update(
            {
                "pack_index": "article.idx",
                "pack_dir": ".",
            }
        )
        return config


class WikiArticleWriter(JsonPackWriter):
    """
    This is a pack writer that writes out the Wikipedia articles on disk. It
    has two special behaviors:
      1. An `article.idx` file will be created at the output directory, it
      maps from the article name to the article path.
      2. The packs are organized into directories. Each directory contains
      at most 2000 documents.

    """

    article_index: TextIO

    # It is difficult to get the type of the csv writer
    # https://stackoverflow.com/questions
    # /51264355/how-to-type-annotate-object-returned-by-csv-writer
    csv_writer: Any

    def __init__(self):
        super().__init__()
        self.article_count: int = 0

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.article_count = 0
        self.article_index = open(
            os.path.join(
                self.configs.output_dir, self.configs.output_index_file
            ),
            "w",
        )
        self.csv_writer = csv.writer(self.article_index, delimiter="\t")

    def sub_output_path(self, pack: DataPack) -> str:
        sub_dir = str(int(self.article_count / 2000)).zfill(5)
        pid = pack.get_single(WikiPage).page_id
        doc_name = f"doc_{self.article_count}" if pid is None else pid
        suffix = ".json.gz" if self.zip_pack else ".json"
        return os.path.join(sub_dir, doc_name) + suffix

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
        super()._process(input_pack)

        out_path = self.sub_output_path(input_pack)
        # Write the index
        self.csv_writer.writerow([input_pack.pack_name, out_path])
        self.article_count += 1

        if self.article_count % 1000 == 0:
            logging.info(
                "Written %s to %s", self.article_count, self.configs.output_dir
            )

    def finish(self, _: Resources):
        self.article_index.close()

    @classmethod
    def default_configs(cls):
        """
        This defines a basic config structure for the reader.

        Here:
          - pack_dir: the directory that contains all the serialized packs.
          - pack_index: the file name under the pack directory that points to
            the index from the name to the actual pack path.

        :return:
        """
        config = super().default_configs()
        config.update(
            {
                "output_index_file": "article.idx",
            }
        )
        return config


class WikiStructReader(WikiPackReader):
    """
    This reader extends the WikiPackReader and add the Wikipedia Structure
    information from https://wiki.dbpedia.org/downloads-2016-10#p10608-2
    """

    def add_wiki_info(self, pack: DataPack, statements: List):
        for nif_range, rel, struct_type in statements:
            r = get_resource_fragment(rel)
            if r is not None and r == "type":
                range_ = get_resource_attribute(nif_range, "char")
                if range_ is None:
                    continue

                begin, end = [int(d) for d in range_.split(",")]

                if end > len(pack.text):
                    # Some nif dataset are off by a bit, mostly when there
                    # are new line characters, we cannot correct them.
                    # but we need to make sure they don't go longer than
                    # the text.
                    logging.info(
                        "NIF Structure end is %d by %s, "
                        "clipped to fit with the text.",
                        end,
                        nif_range,
                    )
                    end = len(pack.text)

                if end <= begin:
                    logging.info(
                        "Provided struct [%d:%d] is invalid.", begin, end
                    )
                    continue

                struct_ = get_resource_fragment(struct_type)

                if struct_ is not None:
                    if struct_ == "Section":
                        WikiSection(pack, begin, end)
                    elif struct_ == "Paragraph":
                        WikiParagraph(pack, begin, end)
                    elif struct_ == "Title":
                        WikiTitle(pack, begin, end)
                    else:
                        logging.warning("Unknown struct type: %s", struct_type)


class WikiAnchorReader(WikiPackReader):
    """
    This reader extends the WikiPackReader and add the Wikipedia anchors
    from https://wiki.dbpedia.org/downloads-2016-10#p10608-2
    """

    def add_wiki_info(self, pack: DataPack, statements: List):
        link_grouped: DefaultDict[
            str, Dict[str, rdflib.term.Node]
        ] = defaultdict(dict)
        for nif_range, rel, info in statements:
            range_ = get_resource_attribute(nif_range, "char")
            r = get_resource_fragment(rel)
            if range_ is not None and r is not None:
                link_grouped[range_][r] = info

        for range_, link_infos in link_grouped.items():
            begin, end = [int(d) for d in range_.split(",")]

            if end > len(pack.text):
                # Some nif dataset are off by a bit, mostly when there are
                # new line characters, we cannot correct them.
                # but we need to make sure they don't go longer than the
                # text.
                logging.info(
                    "Provided anchor end is %d, "
                    "clipped to fit with the text.",
                    end,
                )
                end = len(pack.text)

            if end <= begin:
                logging.info("Provided anchor [%d:%d is invalid.]", begin, end)
                continue

            for info_key, info_value in link_infos.items():
                info_value = str(info_value)
                if info_key == "type":
                    anchor_type = get_resource_fragment(info_value)
                    if (
                        not anchor_type == "Phrase"
                        and not anchor_type == "Word"
                    ):
                        logging.warning("Unknown anchor type: %s", info_value)
                if info_key == "taIdentRef":
                    target_page_name = get_resource_name(info_value)
                    if (
                        target_page_name is not None
                        and target_page_name in self._redirects
                    ):
                        target_page_name = self._redirects[target_page_name]

                    if target_page_name is not None:
                        # Only create anchor with proper link.
                        anchor = WikiAnchor(pack, begin, end)
                        anchor.target_page_name = target_page_name
                        # If it is an DBpedia resource, the domain will be
                        # truncated, otherwise it will stay the same, meaning
                        # it is an external link.
                        anchor.is_external = target_page_name == str(info_value)


class WikiPropertyReader(WikiPackReader):
    """
    This reader extends the WikiPackReader and add the Wikipedia raw info
    boxes (also known as properties) from
    https://wiki.dbpedia.org/downloads-2016-10#p10608-2
    """

    def add_wiki_info(self, pack: DataPack, statements: List):
        for _, v, o in statements:
            slot_name = v.toPython()
            slot_value = get_resource_name(o)
            if slot_value is not None:
                info_box = WikiInfoBoxProperty(pack)
                info_box.key = slot_name
                info_box.value = slot_value


class WikiInfoBoxReader(WikiPackReader):
    """
    This reader extends the WikiPackReader and add the Wikipedia cleaned
    info boxes from https://wiki.dbpedia.org/downloads-2016-10#p10608-2
    """

    def add_wiki_info(self, pack: DataPack, info_box_statements: List):
        for _, v, o in info_box_statements:
            name = get_resource_name(o)
            if name is not None:
                info_box = WikiInfoBoxMapped(pack)
                info_box.key = v.toPython()
                info_box.value = name
