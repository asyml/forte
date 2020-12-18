import os
import tempfile
import unittest
import sqlite3

from forte.data.data_pack import DataPack
from forte.data.readers.stave_readers import StaveMultiDocSqlReader, \
    StaveDataPackSqlReader
from forte.pipeline import Pipeline
from forte.data.data_utils import maybe_download


class StaveReaderTest(unittest.TestCase):
    def setUp(self):
        sql_url = "https://raw.githubusercontent.com/asyml/stave/master" \
                  "/simple-backend/example_db.sql"

        self.datapack_table_name = StaveMultiDocSqlReader.default_configs()[
            'data_pack_table_name']
        self.multipack_table_name = StaveMultiDocSqlReader.default_configs()[
            'multi_pack_table_name']

        self.temp_dir = tempfile.TemporaryDirectory()
        maybe_download(sql_url, self.temp_dir.name, 'example_db.sql')
        sql_script = os.path.join(self.temp_dir.name, 'example_db.sql')
        self.sql_db = os.path.join(self.temp_dir.name, 'db.sqlite3')

        pack_count: int
        mp_count: int

        with open(sql_script) as q_file:
            # Build the example database by executing the sample sql script.
            q = q_file.read()
            conn = sqlite3.connect(self.sql_db)
            c = conn.cursor()
            c.executescript(q)
            conn.commit()

    def _query(self, q: str):
        conn = sqlite3.connect(self.sql_db)
        c = conn.cursor()
        return c.execute(q)

    # TODO: currently only created the pack reader test. Will add multi pack
    #  tests when the database format is determined.
    def test_stave_reader(self):
        # Read number of lines in both tables. These are the number of
        #  packs in each format.
        pack_count = self._query(
            f"SELECT Count(*) FROM {self.datapack_table_name}")

        # Read the data packs using the reader.
        nlp: Pipeline[DataPack] = Pipeline[DataPack]()
        nlp.set_reader(StaveDataPackSqlReader(), config={
            "stave_db_path": self.sql_db
        })
        nlp.initialize()

        read_pack_count = 0
        for _ in nlp.process_dataset():
            read_pack_count += 1

        self.assertEqual(pack_count, read_pack_count)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
