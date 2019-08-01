from typing import Iterator

from nlp.pipeline.data import DataPack
from nlp.pipeline.base_pipeline import BasePipeline
from nlp.pipeline.processors import BaseProcessor, BatchProcessor

__all__ = [
    "Pipeline"
]


class Pipeline(BasePipeline):
    """
    The pipeline consists of a list of predictors.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.initialize(**kwargs)

    def add_processor(self, processor: BaseProcessor):
        if self._ontology is not None:
            processor.set_ontology(self._ontology)
        self._processors_index[processor.component_name] = len(self.processors)
        self.processors.append(processor)

    def process(self, data: str) -> DataPack:
        """
        Process a string text or a single file.

        Args:
            data (str): the path to a file a string text. If :attr:`_reader` is
                :class:`StringReader`, `data` should be a text in the form of
                a string variable. If :attr:`_reader` is a file reader, `data`
                should be the path to a file.
        """
        datapack = self._reader.read(data)

        for processor in self.processors:
            if isinstance(processor, BatchProcessor):
                processor.process(datapack, tail_instances=True)
            else:
                processor.process(datapack)
        return datapack

    def process_dataset(self, dataset: str) -> Iterator[DataPack]:
        """
        Process the documents in the dataset and return an iterator of DataPack.

        Args:
            dataset (str): the directory of the dataset to be processed.

        """

        data_iter = self._reader.dataset_iterator(dataset)

        for pack in data_iter:
            self.current_packs.append(pack)
            for i, processor in enumerate(self.processors):
                for c_pack in self.current_packs:
                    in_cache = (c_pack.meta.cache_state ==
                                processor.component_name)
                    can_process = (i == 0 or c_pack.meta.process_state ==
                                   self.processors[i - 1].component_name)
                    if can_process and not in_cache:
                        processor.process(c_pack)
            for c_pack in list(self.current_packs):
                # must iterate through a copy of the originial list
                # because of the removing operation
                if (c_pack.meta.process_state ==
                        self.processors[-1].component_name):
                    yield c_pack
                    self.current_packs.remove(c_pack)

        # process tail instances in the whole dataset
        for c_pack in list(self.current_packs):
            start = self._processors_index[c_pack.meta.process_state] + 1
            for processor in self.processors[start:]:
                if isinstance(processor, BatchProcessor):
                    processor.process(c_pack, tail_instances=True)
                else:
                    processor.process(c_pack)
            yield c_pack
            self.current_packs.remove(c_pack)
