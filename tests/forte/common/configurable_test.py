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
import unittest
from abc import ABC
from typing import Dict, Any, Optional, Union

from forte.common.configurable import Configurable
from forte.common.configuration import Config


class BaseClassA(ABC):
    def __init__(self):
        self.dummy_variable = "dummy"


class Parent(BaseClassA, Configurable):
    def __init__(self, configs: Optional[Union[Config, Dict[str, Any]]] = None):
        super().__init__()
        self.configs = self.make_configs(configs)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {
            "config1": {
                "level1_key1": "key1_parent_value",
                "level1_key2": "key2_parent_value",
                "level1_key3": {"level2_key": "nested_parent_value"},
            }
        }


class ChildA(Parent):
    def __init__(self, configs: Optional[Union[Config, Dict[str, Any]]] = None):
        super().__init__(configs)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {"config2": "value_a"}


class ChildB(Parent):
    def __init__(self, configs: Optional[Union[Config, Dict[str, Any]]] = None):
        super().__init__(configs)

    @classmethod
    def default_configs(cls) -> Dict[str, Any]:
        return {"config1": {"level1_key3": {"level2_key": "value_b"}}}


class ConfigurableTest(unittest.TestCase):
    def test_config_classes(self):
        parent = Parent()
        self.assertDictEqual(
            parent.configs.todict(),
            {
                "config1": {
                    "level1_key1": "key1_parent_value",
                    "level1_key2": "key2_parent_value",
                    "level1_key3": {"level2_key": "nested_parent_value"},
                }
            },
        )

        parent1 = Parent(
            {
                "config1": {
                    "level1_key3": {"level2_key": "updated_nested_parent_value"}
                }
            }
        )

        self.assertDictEqual(
            parent1.configs.todict(),
            {
                "config1": {
                    "level1_key1": "key1_parent_value",
                    "level1_key2": "key2_parent_value",
                    "level1_key3": {
                        "level2_key": "updated_nested_parent_value"
                    },
                }
            },
        )

        child_a = ChildA()
        self.assertEqual(
            child_a.configs.todict(),
            {
                "config1": {
                    "level1_key1": "key1_parent_value",
                    "level1_key2": "key2_parent_value",
                    "level1_key3": {"level2_key": "nested_parent_value"},
                },
                "config2": "value_a",
            },
        )

        child_a2 = ChildA(
            {
                "config1": {
                    "level1_key1": "updated_value",
                },
                "config2": "value_b",
            }
        )

        self.assertEqual(
            child_a2.configs.todict(),
            {
                "config1": {
                    "level1_key1": "updated_value",
                    "level1_key2": "key2_parent_value",
                    "level1_key3": {"level2_key": "nested_parent_value"},
                },
                "config2": "value_b",
            },
        )

        child_b = ChildB()
        self.assertEqual(
            child_b.configs.todict(),
            {
                "config1": {
                    "level1_key1": "key1_parent_value",
                    "level1_key2": "key2_parent_value",
                    "level1_key3": {"level2_key": "value_b"},
                }
            },
        )

        self.assertEqual(child_b.dummy_variable, "dummy")
