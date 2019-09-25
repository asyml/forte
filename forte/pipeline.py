import logging
from typing import Dict

import yaml
from texar.torch.hyperparams import HParams

from forte.base_pipeline import BasePipeline
from forte.data import DataPack
from forte.utils import get_class

logger = logging.getLogger(__name__)

__all__ = [
    "Pipeline"
]


class Pipeline(BasePipeline[DataPack]):
    """
    The pipeline consists of a list of predictors.
    """
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
                           "nlp.forte.data.ontology"]
            self.set_ontology(get_class(configs["Ontology"], module_path))

        else:
            logger.warning("Ontology not specified in config, will use "
                           "base_ontology by default.")
