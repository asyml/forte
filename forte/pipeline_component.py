from texar.torch import HParams

from forte.common import Resources


class PipeComponent:
    def initialize(self, resource: Resources, configs: HParams):
        """
        The pipeline will call the initialize method at the start of a
        processing. The processor will be initialized with ``configs``,
        and register global resources into ``resource``. The implementation
        should set up the states of the component.

        :param configs: The configuration passed in to set up this component.
        :param resource: A global resource register. User can register
        shareable resources here, for example, the vocabulary.
        :return:
        """

        raise NotImplementedError("This component has not been implemented")

    def finish(self, resource: Resources):
        """
        The pipeline will calls this function at the end of the pipeline to
        notify all the components. The user can implement this function to
        release resources used by this component.

        The component can also add objects to the resources.

        Args:
            resource:

        Returns:

        """
        raise NotImplementedError("This component has not been implemented")
