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
Unit tests for Deserialize Reader.
"""
import os
import sqlite3
import tempfile
import unittest

from ddt import data, ddt

from forte.data.data_pack import DataPack
from forte.data.readers import StringReader, RawDataDeserializeReader, \
    MultiPackSentenceReader
from forte.data.readers.deserialize_reader import dump_packs_from_database
from forte.pipeline import Pipeline


class DeserializeReaderPipelineTest(unittest.TestCase):

    def setUp(self):
        # Define and config the Pipeline
        self.nlp: Pipeline[DataPack] = Pipeline[DataPack]()
        self.nlp.set_reader(StringReader())
        self.nlp.initialize()

    def test_direct_deserialize(self):
        another_pipeline = Pipeline[DataPack]()
        another_pipeline.set_reader(RawDataDeserializeReader())
        another_pipeline.initialize()

        data = ["Testing Reader", "Testing Deserializer"]

        for pack in self.nlp.process_dataset(data):
            for new_pack in another_pipeline.process_dataset(
                    [pack.serialize()]):
                self.assertEqual(pack.text, new_pack.text)


@ddt
class DatabaseLoadingTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

        self.db_path = os.path.join(self.test_dir, 'db.sqlite3')

        sql_create_mp_table = """ CREATE TABLE IF NOT EXISTS 
        nlpviewer_backend_crossdoc (
            id integer PRIMARY KEY,
            name varchar(200) NOT NULL,
            textPack text NOT NULL,
            ontology text NOT NULL
        ); """

        sql_create_pack_table = """ CREATE TABLE IF NOT EXISTS 
        nlpviewer_backend_document (
            id integer PRIMARY KEY,
            name varchar(200) NOT NULL,
            textPack text NOT NULL,
            ontology text NOT NULL
        ); """

        self.conn = sqlite3.connect(self.db_path)
        if self.conn is not None:
            self.conn.cursor().execute(sql_create_mp_table)
            self.conn.cursor().execute(sql_create_pack_table)

        onto_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 *([os.path.pardir] * 4),
                                                 'forte', 'ontology_specs',
                                                 'base_ontology.json'))

        with open(onto_path) as onto:
            self.onto_text = onto.read()

    def add_data(self, table_name, name, pack_str):
        sql = f''' INSERT INTO {table_name}(name, textPack, ontology) 
                    VALUES(?,?,?)'''
        cur = self.conn.cursor()
        cur.execute(sql, (name, pack_str, self.onto_text))
        self.conn.commit()

    @data("This file is used for testing MultiPackSentenceReader.",
          "This tool is called Forte.\n",
          "The goal of this project to help you build NLP pipelines.\n"
          "NLP has never been made this easy before.")
    def test_database_load(self, text):
        file_path = os.path.join(self.test_dir, 'test.txt')
        with open(file_path, 'w') as f:
            f.write(text)

        pl = Pipeline()
        pl.set_reader(MultiPackSentenceReader(), {"suffix": ".txt"})
        pl.initialize()

        multipack = pl.process(self.test_dir)
        src_pack = multipack.get_pack('input_src')
        tgt_pack = multipack.get_pack('output_tgt')

        self.add_data('nlpviewer_backend_crossdoc', 'multi',
                      multipack.serialize())
        self.add_data('nlpviewer_backend_document', 'pack1',
                      src_pack.serialize())
        self.add_data('nlpviewer_backend_document', 'pack2',
                      tgt_pack.serialize())

        dump_packs_from_database(self.db_path, self.test_dir)
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'packs')))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'multi')))

        self.assertTrue(os.path.exists(
            os.path.join(self.test_dir, 'multi', 'multi.json')))
        self.assertTrue(os.path.exists(
            os.path.join(self.test_dir, 'packs', 'pack1.json')))
        self.assertTrue(os.path.exists(
            os.path.join(self.test_dir, 'packs', 'pack2.json')))


if __name__ == '__main__':
    unittest.main()
