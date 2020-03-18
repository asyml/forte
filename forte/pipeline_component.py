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
from typing import Generic

from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.base_pack import PackType


class PipelineComponent(Generic[PackType]):

    # pylint: disable=attribute-defined-outside-init
    def initialize(self, resources: Resources, configs: HParams):
        r"""The pipeline will call the initialize method at the start of a
        processing. The processor and reader will be initialized with
        ``configs``, and register global resources into ``resource``. The
        implementation should set up the states of the component.

        Args:
            resources (Resources): A global resource register. User can register
                shareable resources here, for example, the vocabulary.
            configs (HParams): The configuration passed in to set up this
                component.
        """
        self.resources: Resources = resources
        self.configs: HParams = configs

    def finish(self, resource: Resources):
        r"""The pipeline will call this function at the end of the pipeline to
        notify all the components. The user can implement this function to
        release resources used by this component. The component can also add
        objects to the resources.

        Args:
            resource (Resources): A global resource registry.
        """
        pass

    @staticmethod
    def default_configs():
        r"""Returns a `dict` of configurations of the component with default
        values. Used to replace the missing values of input `configs`
        during pipeline construction.


        """
        return {
        }
