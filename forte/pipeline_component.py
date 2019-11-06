from typing import Generic

from texar.torch import HParams

from forte.common.resources import Resources
from forte.data.base_pack import PackType


class PipelineComponent(Generic[PackType]):
    def initialize(self, resource: Resources, configs: HParams):
        r"""The pipeline will call the initialize method at the start of a
        processing. The processor and reader will be initialized with
        ``configs``, and register global resources into ``resource``. The
        implementation should set up the states of the component.

        Args:
            resource (Resources): A global resource register. User can register
                shareable resources here, for example, the vocabulary.
            configs (HParams): The configuration passed in to set up this
                component.

        Returns:
        """
        pass

    def finish(self, resource: Resources):
        r"""The pipeline will call this function at the end of the pipeline to
        notify all the components. The user can implement this function to
        release resources used by this component.

        The component can also add objects to the resources.

        Args:
            resource (Resources): A global resource registry.

        Returns:

        """
        pass
