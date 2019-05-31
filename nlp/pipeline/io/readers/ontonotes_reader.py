"""
The reader that read ontonotes data into our internal json data format.
"""

from nlp.pipeline.io.readers.base_reader import BaseReader


class OntonotesReader(BaseReader):
    def __init__(self):
        super().__init__()
