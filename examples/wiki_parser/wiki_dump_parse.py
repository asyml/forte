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
from forte.common.types import PackType
from forte.pipeline import Pipeline
from forte.processors.base.writers import JsonPackWriter
from forte.data.readers.wikipedia.dump_reader import WikiDumpReader


class WikiArticleWriter(JsonPackWriter):
    article_index: TextIO

    # It is difficult to get the type of the csv writer
    # https://stackoverflow.com/questions
    # /51264355/how-to-type-annotate-object-returned-by-csv-writer
    csv_writer: Any

    def __init__(self):
        super().__init__()
        self.article_count: int = 0

    def initialize(self, configs: HParams, resources: Resources):
        super(WikiArticleWriter, self).initialize(configs, resources)
        self.article_count = 0
        self.article_index = open(
            os.path.join(self.root_output_dir, 'article.idx'), 'w')
        self.csv_writer = csv.writer(self.article_index, delimiter='\t')

    def sub_output_dir(self, pack: PackType) -> str:
        sub_dir = str(self.article_count % 2000).zfill(5)
        doc_id = str(pack.meta.doc_id)
        return os.path.join(sub_dir, doc_id)

    def process(self, input_pack: PackType):
        """
        Write an index from the document id to the relative storage of this
        DataPack. This can be used as a simple index to retrieve the relevant
        file, which can enable faster lookup in use cases like following the
        Wikipedia links.

        Args:
            input_pack: The DataPack that contains the Wikipedia information.

        Returns:
        """
        super(WikiArticleWriter, self).process(input_pack)
        # TODO: found duplicate entries here.
        self.csv_writer.writerow(
            [input_pack.meta.doc_id, self.sub_output_dir(input_pack)]
        )

    def finish(self, resource: Resources):
        self.article_index.close()


def main(wiki_dump_path: str, output_path: str):
    pl = Pipeline()
    pl.set_reader(WikiDumpReader())

    config = HParams(
        {
            'output_dir': output_path,
            'zip_pack': True,
        },
        WikiArticleWriter.default_hparams()
    )

    pl.add_processor(WikiArticleWriter())

    pl.initialize_processors()
    pl.run(wiki_dump_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
