import os
import logging
import sqlite3
from typing import Dict, Any, Optional, List

from forte.elastic import ElasticSearchIndexer

from forte.common import Resources, ProcessorConfigError
from forte.common.configuration import Config
from forte.data.common_entry_utils import create_utterance, get_last_utterance
from forte.data.data_pack import DataPack
from forte.processors.base import PackProcessor
from ft.onto.base_ontology import Utterance


def sqlite_insert(conn, table, row):
    cols: str = ", ".join('"{}"'.format(col) for col in row.keys())
    vals: str = ", ".join(":{}".format(col) for col in row.keys())
    sql: str = 'INSERT INTO "{0}" ({1}) VALUES ({2})'.format(table, cols, vals)
    cursor = conn.cursor()
    cursor.execute(sql, row)
    conn.commit()
    return cursor.lastrowid


def create_links(url_stub: str, ids: List[int]) -> List[str]:
    links: List[str] = []

    url_stub: str = url_stub.strip("/")
    for temp_idm in ids:
        links.append(
            f"<a href={url_stub}/documents/{temp_idm}>Report #{temp_idm}</a>"
        )
    return links


class LastUtteranceSearcher(PackProcessor):
    # pylint: disable=attribute-defined-outside-init

    def initialize(self, resources: Resources, configs: Config):
        super().initialize(resources, configs)
        self.index = ElasticSearchIndexer(self.configs.indexer.hparams)
        if self.configs.query_result_project_id < 0:
            raise ProcessorConfigError("Query Result Project is not set.")

        if not os.path.exists(self.configs.stave_db_path):
            raise ProcessorConfigError(
                f"Cannot find Stave DB at: {self.configs.stave_db_path}"
            )

    def _process(self, input_pack: DataPack):
        # Make sure we take the last utterance from the user.
        utterance: Optional[Utterance] = get_last_utterance(input_pack, "user")

        if utterance is not None:
            logging.info("The last utterance is %s", utterance)
            # Create the query using the last utterance from user.
            size = self.configs.size or 1000
            field = self.configs.field or "content"
            query_value = {
                "query": {"match": {field: utterance.text}},
                "size": size,
            }

            # Search against the index.
            results = self.index.search(query_value)
            hits = results["hits"]["hits"]

            conn = sqlite3.connect(self.configs.stave_db_path)

            answers = []
            for idx, hit in enumerate(hits):
                source = hit["_source"]
                # The raw pack string and pack id (not database id)
                raw_pack_str: str = source["pack_info"]
                pack_id: str = source["doc_id"]

                # Now you can write the pack into the database and generate url.
                item = {
                    "name": f"clinical_results_{idx}",
                    "textPack": raw_pack_str,
                    "project_id": 5,
                }
                db_id = sqlite_insert(conn, "nlpviewer_backend_document", item)
                answers += [db_id]
                print(pack_id, db_id)

            if len(answers) == 0:
                create_utterance(
                    input_pack,
                    "No results found. Please try another query.",
                    "ai",
                )
            else:
                links: List[str] = create_links(self.configs.url_stub, answers)
                response_text: str = (
                    "I found the following results: <br> -- "
                    + "<br> -- ".join(links)
                )
                print(response_text)

                create_utterance(input_pack, response_text, "ai")
        else:
            logging.info("Cannot get another utterance.")
            create_utterance(
                input_pack,
                "Hey, I didn't get what you say, could you try again?",
                "ai",
            )

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        config = super().default_configs()
        config.update(
            {
                "size": 5,
                "field": "content",
                "indexer": {
                    "name": "ElasticSearchIndexer",
                    "hparams": ElasticSearchIndexer.default_configs(),
                    "other_kwargs": {"request_timeout": 10, "refresh": False},
                },
                "stave_db_path": "~/projects/stave/simple-backend/db.sqlite3",
                "url_stub": "http://localhost:3000",
                "query_result_project_id": -1,
            }
        )
        return config
