"""
Unit tests for reading database samples form Stave.
"""

import importlib
import os
import sqlite3
import sys
import tempfile
import unittest

from ddt import ddt, data

from forte.data.data_pack import DataPack
from forte.data.ontology.ontology_code_generator import OntologyCodeGenerator
from forte.data.readers.stave_readers import (
    StaveMultiDocSqlReader,
    StaveDataPackSqlReader,
)
from forte.pipeline import Pipeline


def query(sql_db, q: str):
    conn = sqlite3.connect(sql_db)
    c = conn.cursor()
    return c.execute(q)


def build_ontology(sql_db, project_name):
    """
    Find the ontology specification from the project, and then create the
    ontologies.

    Args:
        sql_db: The SQLite Database containing the project
        project_name: The name of the project.

    Returns:

    """
    onto_path = "./stave_test_onto"
    res = query(
        sql_db,
        f"SELECT ontology FROM stave_backend_project "
        f"WHERE stave_backend_project.name = "
        f'"{project_name}"',
    ).fetchone()[0]
    with tempfile.NamedTemporaryFile("w") as onto_file:
        onto_file.write(res)
        OntologyCodeGenerator().generate(
            onto_file.name, onto_path, lenient_prefix=True
        )
    # Make sure the newly created path is in the python path.
    sys.path.append(onto_path)

    # Make sure we can import the newly generated modules.
    try:
        importlib.import_module("edu.cmu")
    except Exception:
        pass


@ddt
class StaveReaderTest(unittest.TestCase):
    def setUp(self):
        self.datapack_table: str = StaveMultiDocSqlReader.default_configs()[
            "datapack_table"
        ]
        self.multipack_table: str = StaveMultiDocSqlReader.default_configs()[
            "multipack_table"
        ]
        self.project_table: str = StaveDataPackSqlReader.default_configs()[
            "project_table"
        ]

        # This path correspond to the .travis.yml.
        self.sql_db: str = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                *([os.path.pardir] * 4),
                "db.sqlite3",
            )
        )
        self.assertTrue(os.path.exists(self.sql_db))

    @data("project-1-example", "project-2-example")
    @unittest.skip(
        "The test is skipped because the serialization format in Stave is "
        "outdated. Now we only support deserialization of DataPack whose "
        "pack_version is no less than "
        "``forte.version.PACK_ID_COMPATIBLE_VERSION``."
    )
    # TODO: Regenrate a new serialization string in Stave based on the new
    # implementation with DataStore.
    def test_stave_reader_project(self, project_name: str):
        build_ontology(self.sql_db, project_name)

        # Query packs in this project directly.
        pack_count: int = query(
            self.sql_db,
            f"SELECT Count(*) FROM {self.datapack_table}, {self.project_table} "
            f"WHERE {self.datapack_table}.project_id = {self.project_table}.id "
            f"AND {self.project_table}.name = '{project_name}'",
        ).fetchone()[0]

        # Read the data packs using the reader.
        nlp: Pipeline[DataPack] = Pipeline[DataPack]()
        nlp.set_reader(
            StaveDataPackSqlReader(),
            config={
                "stave_db_path": self.sql_db,
                "target_project_name": project_name,
            },
        )
        nlp.initialize()

        read_pack_count = 0
        for _ in nlp.process_dataset():
            read_pack_count += 1

        self.assertEqual(pack_count, read_pack_count)


if __name__ == "__main__":
    unittest.main()
