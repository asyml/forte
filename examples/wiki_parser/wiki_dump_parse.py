"""
This creates a pipeline to parse the Wikipedia dump and save the results
as MultiPacks onto disk.
"""
import sys
import os
import csv
from typing import TextIO, Any

from texar.torch.hyperparams import HParams

from forte.common.resources import Resources
from forte.data.base_pack import PackType
from forte.data.datasets.wikipedia.dbpedia_based_reader import DBpediaWikiReader
from forte.pipeline import Pipeline
from forte.processors.base.writers import JsonPackWriter

__all__ = [
    'WikiArticleWriter',
]


class WikiArticleWriter(JsonPackWriter):
    article_index: TextIO

    # It is difficult to get the type of the csv writer
    # https://stackoverflow.com/questions
    # /51264355/how-to-type-annotate-object-returned-by-csv-writer
    csv_writer: Any

    def __init__(self):
        super().__init__()
        self.article_count: int = 0

    def initialize(self, resource: Resources, configs: HParams):
        super(WikiArticleWriter, self).initialize(resource, configs)
        self.article_count = 0
        self.article_index = open(
            os.path.join(self.root_output_dir, 'article.idx'), 'w')
        self.csv_writer = csv.writer(self.article_index, delimiter='\t')

    def sub_output_path(self, pack: PackType) -> str:
        sub_dir = str(int(self.article_count / 2000)).zfill(5)
        doc_name = str(pack.meta.doc_id)
        if doc_name is None or doc_name == 'None':
            # Assign default document name based on the count.
            doc_name = f'doc_{self.article_count}'
        return os.path.join(sub_dir, doc_name + '.json')

    def _process(self, input_pack: PackType):
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
        # Write the index
        self.csv_writer.writerow(
            [input_pack.meta.doc_id, self.sub_output_path(input_pack)]
        )
        self.article_count += 1

    def finish(self, resource: Resources):
        # pylint: disable=unused-argument
        self.article_index.close()


def main(nif_context: str, nif_page_structure: str,
         mapping_literals: str, mapping_objects: str,
         nif_text_links: str, output_path: str):
    pl = Pipeline()
    pl.set_reader(DBpediaWikiReader())

    config = HParams(
        {
            'output_dir': output_path,
            'zip_pack': True,
        },
        WikiArticleWriter.default_hparams()
    )

    pl.add_processor(WikiArticleWriter(), config=config)

    pl.initialize()
    pl.run(nif_context, nif_page_structure, mapping_literals, mapping_objects,
           nif_text_links)


if __name__ == '__main__':
    base_dir = sys.argv[1]


    def get_data(dataset: str):
        p = os.path.join(base_dir, dataset)
        if os.path.exists(p):
            return p
        else:
            raise FileNotFoundError(f'The dataset {dataset} is not found in '
                                    f'base directory {base_dir}')


    main(
        get_data('nif_context_en.tql.bz2'),
        get_data('nif_page_structure_en.tql.bz2'),
        get_data('mappingbased_literals_en.tql.bz2'),
        get_data('mappingbased_objects_en.tql.bz2'),
        get_data('nif_text_links_en.tql.bz2'),
        os.path.join(base_dir, 'packs'),
    )
