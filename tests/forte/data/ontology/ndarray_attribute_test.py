import importlib.util
import sys
import os
import tempfile
import unittest
from ddt import data, ddt

import numpy as np
from forte.data.data_pack import DataPack

# import the NdEntry classes manually
module_name = "ft.onto.sample_ndarray"
module_path = os.path.join(os.path.dirname(__file__), "test_outputs/ft/onto/sample_ndarray.py")
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)

NdEntry1 = module.NdEntry1
NdEntry2 = module.NdEntry2
NdEntry3 = module.NdEntry3

globals().update({
    "NdEntry1": module.NdEntry1,
    "NdEntry2": module.NdEntry2,
    "NdEntry3": module.NdEntry3,
})

"""
NdEntry1, NdEntry2, and NdEntry3 are sample Entry containing NdArray attributes
  for testing.
NdEntry1 has both dtype and shape specified,
  while NdEntry2 has only dtype specified and NdEntry3 has only shape specified.
"""


@ddt
class SerializationTest(unittest.TestCase):
    @data(
        NdEntry1,
        NdEntry2,
        NdEntry3
    )
    def test_serialization(self, TestEntry):
        data_pack = DataPack()
        nd_entry = TestEntry(data_pack)
        data_pack.add_entry(nd_entry)

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, "datapack.json")
            data_pack.serialize(output_path, indent=2)

            datapack_deseri = DataPack.deserialize(output_path)
            nd_entry_deseri = datapack_deseri.get_single(TestEntry)

            if nd_entry.value.dtype:
                self.assertEqual(nd_entry.value.dtype, nd_entry_deseri.value.dtype)
            if nd_entry.value.shape:
                self.assertEqual(nd_entry.value.shape, nd_entry_deseri.value.shape)
            if nd_entry.value.data is not None:
                self.assertEqual(np.sum(nd_entry.value.data - nd_entry_deseri.value.data), 0)


@ddt
class PropertyTest(unittest.TestCase):
    @data(
        (NdEntry1, np.array([1], dtype="int")),
        (NdEntry1, np.array([[1, 1], [1, 1]], dtype="float")),
        (NdEntry2, np.array([[1, 1], [1, 1]], dtype="float")),
        (NdEntry3, np.array([1], dtype="int")),
    )
    def test_bad_np_array(self, input_data):
        """
        Test for numpy array with invalid dtype and shape.
        """
        TestEntry, input_array = input_data
        data_pack = DataPack()
        nd_entry = TestEntry(data_pack)
        if nd_entry.value.dtype and input_array.dtype != nd_entry.value.dtype:
            with self.assertRaises(TypeError):
                nd_entry.value.data = input_array

        if nd_entry.value.shape and input_array.shape != nd_entry.value.shape:
            with self.assertRaises(AttributeError):
                nd_entry.value.data = input_array

    @data(
        (NdEntry1, [1]),
        (NdEntry3, [1]),
        (NdEntry1, [[[1]]]),
        (NdEntry3, [[[1]]]),
    )
    def test_bad_py_list(self, input_data):
        """
        Test for python list with invalid shape.
        """
        TestEntry, input_list = input_data
        data_pack = DataPack()
        nd_entry = TestEntry(data_pack)
        input_array = np.array(input_list)
        if nd_entry.value.shape and input_array.shape != nd_entry.value.shape:
            with self.assertRaises(AttributeError):
                nd_entry.value.data = input_list

    @data(
        (NdEntry1, 1),
        (NdEntry2, 1),
        (NdEntry3, 1),
    )
    def test_invalid_input(self, input_data):
        """
        Test for invalid input (anything other than numpy array or python list)
        """
        TestEntry, invalid_value = input_data
        data_pack = DataPack()
        nd_entry = TestEntry(data_pack)
        with self.assertRaises(ValueError):
            nd_entry.value.data = invalid_value

    @data(
        (NdEntry1, [[1, 1], [1, 1]]),
        (NdEntry1, [[1., 1.], [1., 1.]]),
        (NdEntry2, [[1, 1], [1, 1]]),
        (NdEntry2, [1]),
        (NdEntry2, [1.]),
        (NdEntry2, [[1., 1.], [1., 1.]]),
        (NdEntry3, [[1, 1], [1, 1]]),
        (NdEntry3, [[1., 1.], [1., 1.]]),
    )
    def test_valid_py_list(self, input_data):
        TestEntry, input_list = input_data
        data_pack = DataPack()
        nd_entry = TestEntry(data_pack)
        try:
            nd_entry.value.data = input_list
        except Exception:
            self.fail()

    @data(
        (NdEntry1, np.array([[1, 1], [1, 1]], dtype="int")),
        (NdEntry2, np.array([[1, 1], [1, 1]], dtype="int")),
        (NdEntry2, np.array([1, 1], dtype="int")),
        (NdEntry3, np.array([[1, 1], [1, 1]], dtype="int")),
        (NdEntry3, np.array([[1, 1], [1, 1]], dtype="float")),
    )
    def test_valid_np_array(self, input_data):
        TestEntry, input_array = input_data
        data_pack = DataPack()
        nd_entry = TestEntry(data_pack)
        try:
            nd_entry.value.data = input_array
        except Exception:
            self.fail()

        # If assign value successfully, dtype and shape of
        #   nd_entry.value should match to input_array's.
        self.assertEqual(nd_entry.value.dtype, input_array.dtype)
        self.assertEqual(nd_entry.value.shape, input_array.shape)
        self.assertEqual(np.sum(nd_entry.value.data - input_array), 0)
