"""The Dataset class which support getting data or data batch and requesting
entries and fields.
"""
from typing import Iterator, Dict, List, Union, Optional, Iterable
import numpy as np
from nlp.pipeline.data.data_pack import DataPack
from nlp.pipeline.data.base_ontology import BaseOntology


class Dataset:
    """
    Example:
        .. code-block:: python
            reader = OntonotesReader()
            data = reader.dataset_iterator("conll-formatted-ontonotes-5.0")
            dataset = DatasetBase(data)

            annotype = {
                        "Token": ["pos_tag", "sense"],
                        "EntityMention": []
            }
            linktype = {
                "PredicateLink": ["parent", "parent.pred_lemma",
                                  "child", "arg_type"]
            }

            dataset.config_data_iterator(context_type="sentence",
                                         annotation_types = annotype,
                                         link_types = linktype)

            ## Three ways to get data:
            ## 1. get a piece of data
            sentence = dataset.get_data()
            ## 2. get a batch of data
            batch = dataset.get_data_batch(50)
            ## 3. iterate through the dataset
            for sentence in dataset.iterator:
                process(sentence)

    """

    def __init__(self, data_packs):
        self.data_packs: List[DataPack] = data_packs

    def get_data(self):
        for data_pack in self.data_packs:
            for data in data_pack.get_data():
                yield data

    def get_data_batch(self, batch_size: int):

        batch = {}
        for _ in range(batch_size):
            data = self.get_data()
            if data is None:
                break
            for entry, fields in data.items():
                if isinstance(fields, dict):
                    if entry not in batch.keys():
                        batch[entry] = {}
                    for k, value in fields.items():
                        if k not in batch[entry].keys():
                            batch[entry][k] = []
                        batch[entry][k].append(value)
                else:  # context level feature
                    if entry not in batch.keys():
                        batch[entry] = []
                    batch[entry].append(fields)

        if batch:
            return batch
        return None
