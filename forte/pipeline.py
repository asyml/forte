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

import logging
from typing import Dict

import yaml
from texar.torch.hyperparams import HParams

from forte.base_pipeline import BasePipeline
from forte.data.data_pack import DataPack
from forte.utils import get_class, create_class_with_kwargs

logger = logging.getLogger(__name__)

__all__ = [
    "Pipeline"
]


class Pipeline(BasePipeline[DataPack]):
    """
        The main pipeline class for processing DataPack.
    """

    def init_from_config(self, configs: Dict):
        """
        Initialize the pipeline with the configurations

        Args:
            configs: The configurations used to create the pipeline.

        Returns:

        """
        if "Reader" not in configs or configs["Reader"] is None:
            raise KeyError('No reader in the configuration')

        reader_config = configs["Reader"]

        reader, reader_hparams = create_class_with_kwargs(
            class_name=reader_config["type"],
            class_args=reader_config.get("kwargs", {}),
            h_params=reader_config.get("hparams", {}))

        self.set_reader(reader, reader_hparams)

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
                default_processor_hparams = p_class.default_configs()

                processor_hparams = HParams(hparams,
                                            default_processor_hparams)
                self.add_processor(p, processor_hparams)

            self.initialize()
