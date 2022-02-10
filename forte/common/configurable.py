# Copyright 2019-2021 The Forte Authors. All Rights Reserved.
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
from abc import ABC
from typing import Dict, Any, Optional, Union

import yaml

from forte.common import ProcessorConfigError
from forte.common.configuration import Config
from forte.utils import get_full_module_name

__all__ = ["Configurable"]


class Configurable(ABC):
    """
    Classes that implement the Configurable interface enables the classes to
    easily manage configurations. The extended classes should implement
    `default_configs` to provide additional default configurations. A
    `make_configs` functions is provided to create the configurations. There
    are the following behaviors:

      - `default_configs`: the default configs provide the configurations
         structure accepted by this class. This behavior is the same as
         `texar.data.HParams`. If the current class extends another
         `Configurable` class, the `default_configs` of the parent class will
         be inherited in an recursive way. If the same configuration key is
         defined for a parent and child class, the one from the child class
         will prevail.
      - `make_configs`: this function accepts user configurations and create
        the configuration by merging the user config and the default configs.
        Check the docstring of `make_configs` for its behaviors.
    """

    @classmethod
    def _default_configs(cls) -> Config:
        # pylint: disable=protected-access
        merged = Config(cls.default_configs(), {}, allow_new_hparam=True)
        for base in cls.__bases__:
            if hasattr(base, "_default_configs"):
                merged = Config(
                    merged,
                    base._default_configs().todict(),  # type: ignore
                    allow_new_hparam=True,
                )
                break
        return merged

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {}

    @classmethod
    def make_configs(
        cls,
        configs: Optional[Union[Config, Dict[str, Any]]],
    ) -> Config:
        """
        Create the configuration by merging the
        provided config with the `default_configs`.

        The following config conventions are expected:
          - The top level key can be a special `@config_path`.

          - `@config_path` should be point to a file system path, which will
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

            if configs.get("@config_path", None) is not None:
                with open(configs.pop("@config_path"), encoding="utf-8") as f:
                    filebased_configs = yaml.safe_load(f)
            else:
                filebased_configs = {}

            merged_configs.update(filebased_configs)

            merged_configs.update(configs)

        try:
            final_configs = Config(
                merged_configs, cls._default_configs().todict()
            )
        except ValueError as e:
            raise ProcessorConfigError(
                f"Configuration error for the processor "
                f"{get_full_module_name(cls)}."
            ) from e

        return final_configs
