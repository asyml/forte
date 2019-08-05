from abc import abstractmethod
from typing import List, Dict, Iterator, Generic, Optional
import yaml
import logging

from texar.torch import HParams

from nlp.pipeline.utils import get_class
from nlp.pipeline.data.base_pack import PackType
from nlp.pipeline.data.ontology import base_ontology
from nlp.pipeline.data.readers import BaseReader
from nlp.pipeline.processors import BaseProcessor
from nlp.pipeline.common.resources import Resources

logger = logging.getLogger(__name__)


__all__ = [
    "BasePipeline"
]


class BasePipeline(Generic[PackType]):
    """
    The pipeline consists of a list of predictors.
    TODO(Wei): check fields when concatenating processors
    """

    def __init__(self, **kwargs):
        self._reader: BaseReader = None
        self._processors: List[BaseProcessor] = []
        self._processors_index: Dict = {'': -1}
        self._configs: List[Optional[HParams]] = []

        self._ontology = base_ontology
        self.topology = None
        self.current_packs = []
        self.resource = Resources()

        self.initialize(**kwargs)

    def initialize(self, **kwargs):
        """
        Initialize the pipeline with configs
        """
        if "ontology" in kwargs.keys():
            self._ontology = kwargs["ontology"]
            if self._reader is not None:
                self._reader.set_ontology(self._ontology)

    def init_from_config_path(self, config_path):
        """
        Read the configs from the given path ``config_path``
        and initialize the pipeline including processors
        """
        # TODO: Typically, we should also set the reader here
        # This will be done after StringReader is merged
        # We need to modify the read -> read_file_as_pack then.
        configs = yaml.safe_load(open(config_path))

        self.init_from_config(configs)

    def init_from_config(self, configs: Dict):
        """
        Parse the configuration sections from the input config,
            into a list of [processor, config]
        Initialize the pipeline with the configurations
        """

        # HParams cannot create HParams from the inner dict of list

        if "Processors" in configs and configs["Processors"] is not None:

            for processor_configs in configs["Processors"]:

                p_class = get_class(processor_configs["type"])
                if processor_configs.get("kwargs"):
                    processor_kwargs = processor_configs["kwargs"]
                else:
                    processor_kwargs = {}
                p = p_class(**processor_kwargs)

                hparams: Dict = {}

                if processor_configs.get("hparams"):
                    # Extract the hparams section and build hparams
                    processor_hparams = processor_configs["hparams"]

                    if processor_hparams.get("config_path"):
                        filebased_hparams = yaml.safe_load(
                            open(processor_hparams["config_path"]))
                    else:
                        filebased_hparams = {}
                    hparams.update(filebased_hparams)

                    if processor_hparams.get("overwrite_configs"):
                        overwrite_hparams = processor_hparams[
                            "overwrite_configs"]
                    else:
                        overwrite_hparams = {}
                    hparams.update(overwrite_hparams)
                default_processor_hparams = p_class.default_hparams()

                processor_hparams = HParams(hparams,
                                            default_processor_hparams)
                self.add_processor(p, processor_hparams)
                self.initialize_processors()

        if "Ontology" in configs.keys() and configs["Ontology"] is not None:
            module_path = ["__main__",
                           "nlp.pipeline.data.ontology"]
            self._ontology = get_class(configs["Ontology"], module_path)
            for processor in self.processors:
                processor.set_ontology(self._ontology)
        else:
            logger.warning("Ontology not specified in config, will use "
                           "base_ontology by default.")

    def initialize_processors(self):
        for processor, config in zip(self.processors, self.processor_configs):
            processor.initialize(config, self.resource)
            if self._ontology is not None:
                processor.ontology = self._ontology

    def set_reader(self, reader: BaseReader):
        reader.set_ontology(self._ontology)
        self._reader = reader

    @property
    def processors(self):
        return self._processors

    @property
    def processor_configs(self):
        return self._configs

    @abstractmethod
    def add_processor(self,
                      processor: BaseProcessor,
                      config: Optional[HParams] = None):
        raise NotImplementedError

    @abstractmethod
    def process(self, data: str) -> PackType:
        """
        Process a string text or a single file.

        Args:
            data (str): the path to a file a string text. If :attr:`_reader` is
                :class:`StringReader`, `data` should be a text in the form of
                a string variable. If :attr:`_reader` is a file reader, `data`
                should be the path to a file.
        """
        raise NotImplementedError

    @abstractmethod
    def process_dataset(self, dataset: str) -> Iterator[PackType]:
        """
        Process the documents in the dataset and return an iterator of DataPack.

        Args:
            dataset (str): the directory of the dataset to be processed.
        """
        raise NotImplementedError
