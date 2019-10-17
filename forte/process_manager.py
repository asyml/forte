from typing import Optional

__all__ = [
    "ProcessManager"
]


class ProcessManager:
    """
    A pipeline level manager that manages global processing information, such
    as the current running components.
    """

    # Note: hiding the real class creation and attributes here allow us to
    # create a singleton class. A singleton ProcessManager should be sufficient
    # when we are not dealing with multi-process.

    class __ProcessManager:
        def __init__(self):
            self.current_component: str = '__default__'

    instance: Optional[__ProcessManager] = None

    def __init__(self):
        if not ProcessManager.instance:
            ProcessManager.instance = ProcessManager.__ProcessManager()

    def set_current_component(self, component_name: str):
        if self.instance is not None:
            self.instance.current_component = component_name
        else:
            raise AttributeError('The process manager is not initialized.')

    @property
    def component(self):
        return self.instance.current_component
