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
Pipeline component module.
"""
from typing import Generic, Optional, Union, Dict, Any

import yaml
from forte.common import ProcessorConfigError

from forte.common.configuration import Config
from forte.common.resources import Resources
from forte.data.base_pack import PackType
from forte.pack_manager import PackManager
from forte.process_manager import _ProcessManager
from forte.utils import get_full_module_name


class PipelineComponent(Generic[PackType]):
    def __init__(self):
        self._process_manager: _ProcessManager
        self._pack_manager: PackManager = PackManager()

    # pylint: disable=attribute-defined-outside-init
    def assign_manager(self, process_manager: _ProcessManager):
        self._process_manager = process_manager

    def initialize(self, resources: Resources, configs: Config):
        r"""The pipeline will call the initialize method at the start of a
        processing. The processor and reader will be initialized with
        ``configs``, and register global resources into ``resource``. The
        implementation should set up the states of the component.

        Args:
            resources (Resources): A global resource register. User can register
                shareable resources here, for example, the vocabulary.
            configs (Config): The configuration passed in to set up this
                component.
        """
        self.resources: Optional[Resources] = resources
        self.configs: Config = configs

    @property
    def name(self):
        return get_full_module_name(self)

    def flush(self):
        r"""Indicate that there will be no more packs to be passed in, handle
        what's remaining in the buffer."""
        pass

    def finish(self, resource: Resources):
        r"""The pipeline will call this function at the end of the pipeline to
        notify all the components. The user can implement this function to
        release resources used by this component. The component can also add
        objects to the resources.

        Args:
            resource (Resources): A global resource registry.
        """
        pass

    @classmethod
    def make_configs(
            cls, configs: Optional[Union[Config, Dict[str, Any]]]) -> Config:
        """
        Create the component configuration for this class, by merging the
        provided config with the ``default_config``.

        The following config conventions are expected:
          - The top level key can be a special `config_path`.
          - `config_path` should be point to a file system path, which will
             be a YAML file containing configurations.
          - Other key values in the configs will be considered as parameters.

        Args:
            configs: The input config to be merged with the default config.

        Returns:
            The merged configuration.
        """
        merged_configs: Dict = {}

        if configs is not None:
            if isinstance(configs, Config):
                configs = configs.todict()

            if "config_path" in configs and not configs["config_path"] is None:
                filebased_configs = yaml.safe_load(
                    open(configs.pop("config_path")))
            else:
                filebased_configs = {}

            merged_configs.update(filebased_configs)

            merged_configs.update(configs)

        try:
            final_configs = Config(merged_configs, cls.default_configs())
        except ValueError as e:
            raise ProcessorConfigError(
                f'Configuration error for the processor '
                f'{get_full_module_name(cls)}.') from e

        return final_configs

    @classmethod
    def default_configs(cls):
        r"""Returns a `dict` of configurations of the component with default
        values. Used to replace the missing values of input `configs`
        during pipeline construction.
        """
        return {}
