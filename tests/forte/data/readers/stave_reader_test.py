import os
import sys
import tempfile
import unittest
import sqlite3
import importlib

from ddt import ddt, data

from forte.data.data_pack import DataPack
from forte.data.ontology.ontology_code_generator import OntologyCodeGenerator
from forte.data.readers.stave_readers import StaveMultiDocSqlReader, \
    StaveDataPackSqlReader
from forte.pipeline import Pipeline
from forte.data.data_utils import maybe_download


@ddt
class StaveReaderTest(unittest.TestCase):
    def setUp(self):
        sql_url = "https://raw.githubusercontent.com/asyml/stave/master" \
                  "/simple-backend/example_db.sql"

        self.datapack_table = StaveMultiDocSqlReader.default_configs()[
            'datapack_table']
        self.multipack_table = StaveMultiDocSqlReader.default_configs()[
            'multipack_table']
        self.project_table = StaveDataPackSqlReader.default_configs()[
            'project_table'
        ]

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

    @data('project-1-example', 'project-2-example')
    def test_stave_reader_project(self, project_name: str):
        def build_ontology():
            onto_path = "./stave_test_onto"
            res = self._query(f'SELECT ontology FROM nlpviewer_backend_project '
                              f'WHERE nlpviewer_backend_project.name = '
                              f'"{project_name}"').fetchone()[0]
            with tempfile.NamedTemporaryFile('w') as onto_file:
                onto_file.write(res)
                OntologyCodeGenerator().generate(
                    onto_file.name, onto_path, lenient_prefix=True
                )
            # Make sure the newly created path is in the python path.
            sys.path.append(onto_path)

            # Make sure we can import the newly generated modules.
            try:
                importlib.import_module('edu.cmu')
            except Exception:
                pass

        build_ontology()

        # Query packs in this project directly.
        pack_count: int = self._query(
            f"SELECT Count(*) FROM {self.datapack_table}, {self.project_table} "
            f"WHERE {self.datapack_table}.project_id = {self.project_table}.id "
            f"AND {self.project_table}.name = '{project_name}'").fetchone()[0]

        # Read the data packs using the reader.
        nlp: Pipeline[DataPack] = Pipeline[DataPack]()
        nlp.set_reader(StaveDataPackSqlReader(), config={
            "stave_db_path": self.sql_db,
            "target_project_name": project_name
        })
        nlp.initialize()

        read_pack_count = 0
        for _ in nlp.process_dataset():
            read_pack_count += 1

        self.assertEqual(pack_count, read_pack_count)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
