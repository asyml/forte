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
