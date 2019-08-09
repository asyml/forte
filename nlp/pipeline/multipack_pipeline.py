import logging
from typing import Dict, List

import yaml
from texar.torch import HParams

from nlp.pipeline.base_pipeline import BasePipeline
from nlp.pipeline.data import MultiPack
from nlp.pipeline.data.selector import Selector
from nlp.pipeline.processors import BaseBatchProcessor
from nlp.pipeline.utils import get_class

logger = logging.getLogger(__name__)

__all__ = [
    "MultiPackPipeline"
]


# pylint: disable=attribute-defined-outside-init

class MultiPackPipeline(BasePipeline[MultiPack]):
    """
    The pipeline consists of a list of predictors.
    """
    def __init__(self):
        super().__init__()
        self._selectors: List[Selector] = []

    @property
    def selectors(self):
        return self._selectors

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

                selector_hparams = processor_hparams.selector
                selector_class = get_class(selector_hparams['type'])
                selector_kwargs = selector_hparams["kwargs"]
                selector = selector_class(**selector_kwargs)
                self.add_selector(selector)

            self.initialize_processors()

        if "Ontology" in configs.keys() and configs["Ontology"] is not None:
            module_path = ["__main__",
                           "nlp.pipeline.data.ontology"]
            self._ontology = get_class(
                configs["Ontology"],
                module_path)
            for processor in self.processors:
                processor.set_ontology(self._ontology)
        else:
            logger.warning("Ontology not specified in config, will use "
                           "base_ontology by default.")

    def add_selector(self, selector: Selector):
        self._selectors.append(selector)

    def process(self, data: str) -> MultiPack:
        """
        Process a string text or a single file.

        Args:
            data (str): the path to a file a string text. If :attr:`_reader` is
                :class:`StringReader`, `data` should be a text in the form of
                a string variable. If :attr:`_reader` is a file reader, `data`
                should be the path to a file.
        """
        datapack = self._reader.read(data)

        for processor, selector in zip(self.processors, self.selectors):
            input_pack = selector.select(datapack)
            if isinstance(processor, BaseBatchProcessor):
                processor.process(input_pack, tail_instances=True)
            else:
                processor.process(input_pack)
        return datapack
