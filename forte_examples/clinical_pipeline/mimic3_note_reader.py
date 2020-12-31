import csv
import logging
from pathlib import Path
from typing import Any, Iterator, Union, List

from smart_open import open

from demo.clinical import Description, Body
from forte.data.data_pack import DataPack
from forte.data.readers.base_reader import PackReader


class Mimic3DischargeNoteReader(PackReader):
    """This class is designed to read the discharge notes from MIMIC3 dataset
    as plain text packs.

    For more information for the dataset, visit:
      https://mimic.physionet.org/
    """

    def __init__(self):
        super().__init__()
        self.headers: List[str] = []
        self.text_col = -1  # Default to be last column.
        self.description_col = 0  # Default to be first column.
        self.__note_count = 0  # Count number of notes processed.

    def _collect(self, mimic3_path: Union[Path, str]) -> Iterator[Any]:
        with open(mimic3_path) as f:
            for r in csv.reader(f):
                if 0 < self.configs.max_num_notes <= self.__note_count:
                    break
                yield r

    def _parse_pack(self, row: List[str]) -> Iterator[DataPack]:
        if len(self.headers) == 0:
            self.headers.extend(row)
            for i, h in enumerate(self.headers):
                if h == 'TEXT':
                    self.text_col = i
                    logging.info(f"Text Column is {i}")
                if h == 'DESCRIPTION':
                    self.description_col = i
                    logging.info(f"Description Column is {i}")
        else:
            pack: DataPack = DataPack()
            description: str = row[self.description_col]
            text: str = row[self.text_col]
            delimiter = '\n-----------------\n'
            full_text = description + delimiter + text
            pack.set_text(full_text)

            Description(pack, 0, len(description))
            Body(pack, len(description) + len(delimiter), len(full_text))
            self.__note_count += 1
            yield pack

    @classmethod
    def default_configs(cls):
        config = super().default_configs()
        # If this is set (>0), the reader will only read up to
        # the number specified.
        config['max_num_notes'] = -1
        return config
