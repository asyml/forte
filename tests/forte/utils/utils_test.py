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
Unit test for utilities.
"""
import unittest

from forte.data.readers import StringReader
from forte.pipeline import Pipeline
from forte.utils import utils


class UtilsTest(unittest.TestCase):

    def test_get_full_module_name(self):
        from forte.processors.lowercaser_processor import LowerCaserProcessor
        full_name = utils.get_full_module_name(LowerCaserProcessor)
        self.assertEqual(
            full_name,
            'forte.processors.lowercaser_processor.LowerCaserProcessor')

    def test_get_class_name(self):
        from forte.processors.lowercaser_processor import LowerCaserProcessor
        processor: LowerCaserProcessor = LowerCaserProcessor()
        self.assertEqual(utils.get_class_name(processor), 'LowerCaserProcessor')
        self.assertEqual(utils.get_class_name(processor, True),
                         'lowercaserprocessor')

    def test_get_class(self):
        cls = utils.get_class('LowerCaserProcessor',
                              ['forte.processors.lowercaser_processor'])
        self.assertEqual(cls.__name__, 'LowerCaserProcessor')

        with self.assertRaises(ValueError):
            utils.get_class('NonExistProcessor')

        with self.assertRaises(ValueError):
            utils.get_class('NonExistProcessor', [
                'forte.processors.lowercaser_processor'])

    def test_get_qual_name(self):
        from forte.processors.lowercaser_processor import LowerCaserProcessor
        processor: LowerCaserProcessor = LowerCaserProcessor()

        self.assertEqual(
            utils.get_qual_name(processor),
            'LowerCaserProcessor')

    def test_create_class_with_kwargs(self):
        p, hparams = utils.create_class_with_kwargs(
            class_name=
            'forte.processors.lowercaser_processor.LowerCaserProcessor',
            class_args={},
            h_params={
                'overwrite_configs': {
                    'selector': {
                        'type': 'forte.data.selector.AllPackSelector',
                        'args': None,
                        'kwargs': {}
                    }
                }
            }
        )

        self.assertEqual(
            p.component_name,
            'forte.processors.lowercaser_processor.LowerCaserProcessor')

        # Note that here the provided hparam is merged with the
        # default config, but not assigned to the processor yet.
        self.assertEqual(
            hparams.selector.type,
            'forte.data.selector.AllPackSelector'
        )


if __name__ == '__main__':
    unittest.main()
