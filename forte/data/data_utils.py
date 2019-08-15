import os
from typing import Iterator


class DataUtils:

    def __init__(self, file_extension=""):
        self.file_extension = file_extension

    def dataset_path_iterator(self, dir_path: str) -> Iterator[str]:
        """
        An iterator returning file_paths in a directory containing files
        of the given format
        """
        for root, _, files in os.walk(dir_path):
            for data_file in files:
                if data_file.endswith(self.file_extension):
                    yield os.path.join(root, data_file)
