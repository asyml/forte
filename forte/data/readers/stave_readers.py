# Copyright 2020 The Forte Authors. All Rights Reserved.
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
This class contains readers to read from the Stave annotation tool.

The Stave annotation tool can be found here: https://github.com/asyml/stave
"""
import logging
import sqlite3
from typing import Iterator, Dict, Optional

from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data.data_pack import DataPack
from forte.data.base_reader import PackReader
from forte.data.readers.deserialize_reader import MultiPackDeserializerBase

__all__ = [
    "StaveMultiDocSqlReader",
    "StaveDataPackSqlReader",
]


def load_all_datapacks(
    conn, pack_table_name: str, pack_col: int
) -> Dict[int, DataPack]:
    """
    Load all the data packs from the table given a sqlite connection to the
      Stave database.

    Args:
        conn:  The sqlite database connection.
        pack_table_name: The name of the pack table.
        pack_col: The column number to retrieve the actual pack content.

    Returns:
        A dictionary contains all the datapacks.

    """
    c = conn.cursor()
    data_packs: Dict[int, DataPack] = {}
    for val in c.execute(f"SELECT * FROM {pack_table_name}"):
        pack: DataPack = DataPack.deserialize(val[pack_col])
        # Currently assume we do not have access to the id in the database,
        #  once we update all Stave db format, we can add the real id.
        data_packs[pack.pack_id] = pack
    return data_packs


class StaveMultiDocSqlReader(MultiPackDeserializerBase):
    """
    This reader reads multi packs from Stave's database schema.

    Users of this reader need to first make sure to build the ontology classes
    based on the ontology from the database.

    .. code-block:: python

        pl = Pipeline()
        pl.set_reader(StaveMultiDocSqlReader, config={
            "stave_db_path": self.sql_db,
        })

    The current Stave master does not provide a project key for multi doc, this
    implementation does not use the project key.

    Here, note that each project may have a different ontology system, thus
    users of this project need to make sure the classes correspond to the
    ontology are in place.

    Note: Stave is a annotation interface built on Forte's format:
     - https://github.com/asyml/stave
    """

    def initialize(self, resources: Resources, configs: Config):
        # pylint: disable=attribute-defined-outside-init
        super().initialize(resources, configs)

        if not configs.stave_db_path:
            raise ProcessorConfigError(
                "The database path to stave is not specified."
            )

        self.conn = sqlite3.connect(configs.stave_db_path)
        self.data_packs: Dict[int, DataPack] = load_all_datapacks(
            self.conn, configs.datapack_table, configs.pack_content_col
        )

    def _get_multipack_content(self) -> Iterator[str]:  # type: ignore
        c = self.conn.cursor()
        for value in c.execute(
            f"SELECT textPack FROM {self.configs.multipack_table}"
        ):
            yield value[0]

    def _get_pack_content(self, pack_id: int) -> Optional[DataPack]:
        return self.data_packs.get(pack_id, None)

    @classmethod
    def default_configs(cls):
        """
        The default configurations for this reader, the default value of the
        configurations are as followed:

        .. code-block:: python

            {
                "stave_db_path": None,
                "multipack_table": 'nlpviewer_backend_crossdoc',
                "multipack_content_col": 2,
                "datapack_table": 'nlpviewer_backend_document',
                "pack_content_col": 2,
            }

        Here:

        `"stave_db_path"`: str
            Path to the database file.
        `"multipack_table"`: str
            The table name that contains the multi pack content.
        `"multipack_content_col"`: int
            Indicate the column of the `multipack_table` that contains the
            multi pack content.
        `"datapack_table"`: str
            The table name that contains the data pack content.
        `"pack_content_col"`: str
            The col that contains the data pack content.

        Returns: The default configs of this reader.

        """
        config = super().default_configs()
        config.update(
            {
                "stave_db_path": None,
                "multipack_table": "nlpviewer_backend_crossdoc",
                "multipack_content_col": 2,
                "datapack_table": "nlpviewer_backend_document",
                "pack_content_col": 2,
            }
        )
        return config


class StaveDataPackSqlReader(PackReader):
    """
    This reader reads data packs from Stave's database.

    Users of this reader need to first make sure to build the ontology classes
    based on the ontology from the database.

    .. code-block:: python

        pl = Pipeline()
        pl.set_reader(StaveMultiDocSqlReader, config={
            "stave_db_path": self.sql_db,
            "target_project_name": project_name
        })

    Here, note that each project may have a different ontology system, thus
    users of this project need to make sure the classes correspond to the
    ontology are in place.

    If you are able to use a shared ontology for all projects, you may also
    read all data packs without specifying the project name.

    .. code-block:: python

        pl = Pipeline()
        pl.set_reader(StaveMultiDocSqlReader, config={
            "stave_db_path": self.sql_db
        })


    Note: Stave is a annotation interface built on Forte's format:
     - https://github.com/asyml/stave
    """

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)

        if not configs.stave_db_path:
            raise ProcessorConfigError(
                "The database path to stave is not specified."
            )

        if not configs.datapack_table:
            raise ProcessorConfigError(
                "The table name that stores the data pack is not stored."
            )

        if not configs.target_project_name:
            logging.info(
                "No project specified, will attempt to read all proejcts."
            )

    def _collect(self) -> Iterator[str]:  # type: ignore
        # pylint: disable=attribute-defined-outside-init
        self.conn = sqlite3.connect(self.configs.stave_db_path)
        c = self.conn.cursor()

        pack: str = self.configs.datapack_table
        project: str = self.configs.project_table

        if self.configs.target_project_name is None:
            # Read all documents in the database.
            query = f"SELECT textPack FROM {pack}"
        else:
            # Read the specific project.
            query = (
                f"SELECT textPack FROM {pack}, {project} "
                f"WHERE {pack}.project_id = {project}.id "
                f'AND {project}.name = "{self.configs.target_project_name}"'
            )

        for value in c.execute(query):
            yield value[0]

    def _parse_pack(self, pack_str: str) -> Iterator[DataPack]:
        yield DataPack.deserialize(pack_str)

    @classmethod
    def default_configs(cls):
        """
        The default configurations in this project, the default value of the
        configurations are as followed:

        .. code-block:: python

            {
                "stave_db_path": None,
                "datapack_table": 'nlpviewer_backend_document',
                "pack_content_col": 2,
                "project_table": "nlpviewer_backend_project",
                "target_project_name": None,
            }

        Here:

        `"stave_db_path"`: str
            Path to the database file.
        `"datapack_table"`: str
            The table name that contains the data pack documents.
        `"pack_content_col"`: int
            Indicate the column of the table that contains the data
              pack content.
        `"project_table"`: str
            The table that contains the project information.
        `"target_project_name"`: str
            The table name that contains the project information.

        Returns: The default configs of this reader.

        """
        config = super().default_configs()
        config.update(
            {
                "stave_db_path": None,
                "datapack_table": "nlpviewer_backend_document",
                "pack_content_col": 2,
                "project_table": "nlpviewer_backend_project",
                "target_project_name": None,
            }
        )
        return config
