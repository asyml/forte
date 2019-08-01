from typing import Iterator, List, Optional, Dict, Union
from pydoc import locate
from nlp.pipeline.data import DataPack
from nlp.pipeline.data.readers import BaseReader
from nlp.pipeline.processors import BaseProcessor, BatchProcessor
from nlp.pipeline.utils import get_class
from nlp.pipeline.common.resources import Resources
from texar.torch import HParams
import yaml


class Pipeline:
    """
    The pipeline consists of a list of predictors.
    """

    def __init__(self, **kwargs):
        self._reader: BaseReader = BaseReader()
        self._processors: List[BaseProcessor] = []
        self._processors_index: Dict = {'': -1}

        self._ontology = None
        self.topology = None
        self.current_packs = []

    def init_from_config_path(self, config_path):
        """
        Hparam

        Hprams....
        init_from_config(self, config: HParams):
        :return:
        """
        # TODO: Typically, we should also set the reader here,
        # We need to modify the read -> read_file_as_pack then.
        configs = yaml.safe_load(open(config_path))

        configs = HParams(configs, default_hparams=None)
        resources = Resources().load(configs["Resources"]["storage_path"])

        print(configs)
        for processor in configs.Processors:
            print(type(processor), processor)
            kwargs = getattr(processor, "kwargs")
            p = locate(processor["type"])(**kwargs)
            p.initialize(resources)
            self.add_processor(p)

    def init_from_config(self, config: HParams):
        """
        parse the configuration sections from the input config,
        into a list of [processor, config]
        """
        """
        Initialize the pipeline with configs
        """
        if "ontology" in kwargs.keys():
            module_path = ["__main__",
                           "nlp.pipeline.data.ontology"]
            self._ontology = get_class(kwargs["ontology"], module_path)
            for processor in self.processors:
                processor.ontology = self._ontology

    def set_reader(self, reader: BaseReader):
        self._reader = reader

    @property
    def processors(self):
        return self._processors

    def add_processor(self, processor: BaseProcessor):
        if self._ontology:
            processor.ontology = self._ontology
        self._processors_index[processor.component_name] = len(self.processors)
        self.processors.append(processor)

    def process(self, text: str):
        """
        delegate to reader...

        :param text:
        :return:
        """

        datapack = DataPack()
        datapack.set_text(text)

        for processor in self.processors:
            if isinstance(processor, BatchProcessor):
                processor.process(datapack, tail_instances=True)
            else:
                processor.process(datapack)
        return datapack

    def process_dataset(
            self,
            dataset: Optional[Union[Dict, str]] = None) -> Iterator[DataPack]:
        """
        Process the documents in the dataset and return an iterator of DataPack.

        Args:
            dataset (str or dict, optional): the dataset to be processed. This
                could be a str path to the dataset or a dict including the str
                path and the data format.
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
