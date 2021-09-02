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
from forte.data.base_pack import PackType, BasePack
from forte.data.ontology.core import Entry
from forte.utils import get_full_module_name


class PipelineComponent(Generic[PackType]):
    """
    The base class for all pipeline component. A pipeline component represents
    one node in the pipeline, and would perform certain action on the data
    pack. All pipeline components should extend this class.

    Attributes:
        resources: The resources that can be used by this component, the
          `resources` object is a shared object across the whole pipeline.
        configs: The configuration of this component, will be built by the
          pipeline based on the `default_configs()` and the configs provided
          by the users.
    """

    def __init__(self):
        self.resources: Resources = Resources()
        self.configs: Config = Config({}, {})
        # Determine whether to check the consistencies between components.
        self._check_type_consistency: bool = False
        # The flag indicating whether the component is initialized.
        self.__is_initialized: bool = False

    def enforce_consistency(self, enforce: bool = True):
        r"""This function determines whether the pipeline will enforce
        the content expectations specified in each pipeline component. Each
        component will check whether the input pack contains the expected data
        via checking the meta-data, and throws a
        :class:`~forte.common.exception.ExpectedEntryNotFound` if the check
        fails. When this function is called with enforce is ``True``, all the
        pipeline components would check if the input datapack record matches
        with the expected types and attributes if function
        ``expected_types_and_attributes`` is implemented
        for the processor. For example, processor A requires entry type of
        ``ft.onto.base_ontology.Sentence``, and processor B would
        produce this type in the output datapack, so ``record`` function
        of processor B writes the record of this type in the datapack
        and processor A implements ``expected_types_and_attributes`` to add this
        type. Then when the pipeline runs with enforce_consistency, processor A
        would check if this type exists in the record of the output of the
        previous pipeline component.

        Args:
            enforce: A boolean of whether to enable consistency checking
                for the pipeline or not.
        """
        self._check_type_consistency = enforce

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
        self.resources = resources
        self.configs = configs
        self.__is_initialized = True

    def reset_flags(self):
        """
        Reset the flags related to this component. This will be called first
        when doing initialization.
        """
        self.__is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self.__is_initialized

    def add_entry(self, pack: BasePack, entry: Entry):
        """
        The component can manually call this function to add the entry into
        the data pack immediately. Otherwise, the system will add the entries
        automatically when this component finishes.

        Args:
            pack (BasePack): The pack to add the entry into.
            entry (Entry):  The entry to be added.

        Returns:

        """
        pack.add_entry(entry, self.name)

    @property
    def name(self):
        return get_full_module_name(self)

    def flush(self):
        r"""Indicate that there will be no more packs to be passed in, handle
        what's remaining in the buffer."""
        pass

    def finish(self, resource: Resources):
        # pylint: disable = unused-argument
        r"""The pipeline will call this function at the end of the pipeline to
        notify all the components. The user can implement this function to
        release resources used by this component. The component can also add
        objects to the resources.

        Args:
            resource (Resources): A global resource registry.
        """
        self.__is_initialized = False

    @classmethod
    def make_configs(
        cls, configs: Optional[Union[Config, Dict[str, Any]]]
    ) -> Config:
        """
        Create the component configuration for this class, by merging the
        provided config with the ``default_configs()``.

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
                with open(configs.pop("config_path"), encoding="utf-8") as f:
                    filebased_configs = yaml.safe_load(f)
            else:
                filebased_configs = {}

            merged_configs.update(filebased_configs)

            merged_configs.update(configs)

        try:
            final_configs = Config(merged_configs, cls.default_configs())
        except ValueError as e:
            raise ProcessorConfigError(
                f"Configuration error for the processor "
                f"{get_full_module_name(cls)}."
            ) from e

        return final_configs

    @classmethod
    def default_configs(cls):
        r"""Returns a `dict` of configurations of the component with default
        values. Used to replace the missing values of input `configs`
        during pipeline construction.
        """
        return {}
