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
Utility functions
"""
import sys
import difflib
from functools import wraps, lru_cache
from inspect import getfullargspec
from pydoc import locate
from typing import Dict, List, Optional, get_type_hints, Tuple

from sortedcontainers import SortedList
from typing_inspect import is_union_type, get_origin

__all__ = [
    "get_full_module_name",
    "get_class_name",
    "get_class",
    "get_qual_name",
    "create_class_with_kwargs",
    "check_type",
    "DiffAligner",
    "create_import_error_msg",
]


def get_full_module_name(o, lower: bool = False) -> str:
    r"""Returns the full module and class name of an object ``o``. For example,
    for our :class:`OntonotesReader`, returns
    :class:'forte.data.readers.ontonotes_reader.OntonotesReader'.

    Args:
        o (object): An object class.
        lower (bool): Whether to lowercase the full module and class name.

    Returns:
         The full module and class name.
    """
    if not isinstance(o, type):
        o = o.__class__
    module = o.__module__
    if module is None or module == str.__class__.__module__:
        return o.__name__
    name = module + "." + o.__name__
    if lower:
        return name.lower()
    else:
        return name


def get_class_name(o, lower: bool = False) -> str:
    r"""Returns the class name of an object ``o``.

    Args:
        o (object): An object class.
        lower (bool): Whether to lowercase the class name.

    Returns:
         The class name.
    """
    if not isinstance(o, type):
        o = o.__class__
    if lower:
        return o.__name__.lower()
    else:
        return o.__name__


@lru_cache()
def cached_locate(name_to_locate_class):
    r"""Wrapped version of locate for loading class based on
        ``name_to_locate_class``, cached by lru_cache.
    Args:
        name_to_locate_class (str): Name or full path to the class.

    Returns:
        The target class.

    """
    return locate(name_to_locate_class)


def get_class(
    full_class_name: str,
    module_paths: Optional[List[str]] = None,
    cached_lookup: Optional[bool] = True,
):
    r"""Returns the class based on class name, with cached lookup option.

    Args:
        full_class_name (str): Name or full path to the class.
        module_paths (list): Paths to candidate modules to search for the
            class. This is used if the class cannot be located solely based on
            ``class_name``. The first module in the list that contains the class
            is used.
        cached_lookup (bool): Flag to use "cached_locate" for class loading

    Returns:
        The target class.

    """
    if cached_lookup:
        class_ = cached_locate(full_class_name)
    else:
        class_ = locate(full_class_name)
    if (class_ is None) and (module_paths is not None):
        for module_path in module_paths:
            if cached_lookup:
                class_ = cached_locate(".".join([module_path, full_class_name]))
            else:
                class_ = locate(".".join([module_path, full_class_name]))
            if class_ is not None:
                break

    # Try to find classes that are dynamically loaded, class_ will still be None if failed.
    if class_ is None:
        try:
            module_name, class_name = full_class_name.rsplit(".", 1)
            try:
                class_ = getattr(sys.modules[module_name], class_name)
            except (AttributeError, KeyError):
                # ignore when cannot find the module in sys.modules or cannot find the class
                # in the module.
                pass
        except ValueError:
            # ignoring when the full class name doesn't have multiple parts.
            pass

    if class_ is None:
        if module_paths:
            raise ValueError(
                f"Class not found in {module_paths}: {full_class_name}"
            )
        else:
            raise ValueError(f"Class not found in {full_class_name}")

    return class_


def get_qual_name(o: object, lower: bool = False) -> str:
    r"""Returns the qualified name of an object ``o``.

    Args:
        o: An object class.
        lower: Whether to lowercase the qualified class name.

    Returns:
         The qualified class name.
    """
    if not isinstance(o, type):
        o = o.__class__
    if lower:
        return o.__qualname__.lower()
    else:
        return o.__qualname__


def create_class_with_kwargs(class_name: str, class_args: Dict):
    r"""Create class with the given arguments.

    Args:
        class_name: Class name.
        class_args: Class arguments.

    Returns:
        An object with class of type `class_name`.
    """
    cls = get_class(class_name)
    if not class_args:
        class_args = {}
    obj = cls(**class_args)

    return obj


def check_type(obj, tp):
    if is_union_type(tp):
        return any(check_type(obj, a) for a in tp.__args__)
    else:
        origin = get_origin(tp)
        if origin is None or origin == tp:
            return isinstance(obj, tp)
        else:
            return check_type(obj, origin)


def validate_input(func, **kwargs):
    hints = get_type_hints(func)

    # iterate all type hints
    for attr_name, attr_type in hints.items():
        if attr_name == "return":
            continue

        if not isinstance(kwargs[attr_name], attr_type):
            raise TypeError(
                f"{attr_name} should be of type {attr_type}, "
                f"got type {type(kwargs[attr_name])}"
            )


def type_check(func):
    @wraps(func)
    def wrapped_decorator(*args, **kwargs):
        # translate *args into **kwargs
        func_args = getfullargspec(func)[0]
        kwargs.update(dict(zip(func_args, args)))
        validate_input(func, **kwargs)
        return func(**kwargs)

    return wrapped_decorator


class DiffAligner:
    def __init__(self):
        self.__matcher = difflib.SequenceMatcher(autojunk=False)
        self.__begin_mapper = DiffAligner.OffsetMapper()
        self.__end_mapper = DiffAligner.OffsetMapper()

    class OffsetMapper:
        def __init__(self):
            self.ref_points = SortedList()
            self.adjustments = []

        def set_ref_points(self, p: int, adjustment: int):
            self.ref_points.add(p)
            self.adjustments.append(adjustment)

        def trans_offset(self, origin: int) -> int:
            """
            Translate the offset based on the reference points. If there is no
            translation found, will return None.

            Args:
                origin: The original offset

            Returns:
                The translated offset.

            """
            idx = self.ref_points.bisect_left(origin)
            if idx >= len(self.adjustments):
                return origin + self.adjustments[-1]
            return origin + self.adjustments[idx]

        def clear(self):
            self.ref_points.clear()
            self.adjustments.clear()

    def align_with_segments(
        self, text: str, segments: List[str]
    ) -> List[Optional[Tuple[int, int]]]:
        """
        Provided a text sequence `text`, and a list of text `segments`,
        this function will try to align the `text` with the `segments`,
        and return the best guess on how each segment is mapped to the
        original text. The return value will contain the offset of each
        segment, after mapping on the original text. The guess process is
        based on the "SequenceMatcher" in difflib of Python.

        Calling this function will create a mapping stored inisde the class.

        This is useful as a approximated way to map a text segmentation that
        does not contain offset information to the original text. For example,

        >>> aligner = DiffAligner()
        >>> aligner.align_with_segments("aa bb   cc", ["aa", "bb", "cc"])
        [(0, 2), (3, 5), (6, 10)]

        Args:
            text: The original text.
            segments: The list of segments to be mapped.

        Returns:
            The list of mapped offsets for each segment.
        """
        aligned_spans: List[Optional[Tuple[int, int]]] = []
        segment_spans: List[Tuple[int, int]] = []

        # Construct the text using the segments, and store position and
        # length of each element.
        segment_text: str = ""
        offset = 0
        for s in segments:
            segment_text += s
            segment_text += " "
            segment_spans.append((offset, offset + len(s)))
            offset += len(s) + 1

        self.build_alignment(text.strip(), segment_text)

        for ss in segment_spans:
            b, e = self.trans_span(ss)
            if b >= e or b >= len(text) or e > len(text):
                aligned_spans.append(None)
            else:
                aligned_spans.append((b, e))

        return aligned_spans

    def trans_span(self, span: Tuple[int, int]) -> Tuple[int, int]:
        """
        Translate the provided span based on the alignment computed. Need to
        run after `build_alignment`.

        Args:
            span: The span to be translated.

        Returns:
            The translated span.
        """
        return (
            self.__begin_mapper.trans_offset(span[0]),
            self.__end_mapper.trans_offset(span[1]),
        )

    def clear_alignment(self):
        self.__begin_mapper.clear()
        self.__end_mapper.clear()

    def build_alignment(self, text: str, text_to_align: str):
        # Compute character level mappings.
        self.__matcher.set_seqs(text, text_to_align)
        self.clear_alignment()

        for _, i1, i2, j1, j2 in self._get_opcodes():
            self.__begin_mapper.set_ref_points(j1, i1 - j1)
            self.__end_mapper.set_ref_points(j2, i2 - j2)

    def _get_opcodes(self):
        yield from self.__matcher.get_opcodes()


def create_import_error_msg(
    extra_module: str,
    forte_module: str,
    component_name: str,
    pip_installable: bool = True,
):
    """
    Create an error message for importing package extra required by a forte
    module.


    Args:
        extra_module: module name should be installed by pip.
        forte_module: forte module User should install by
            ``pip install forte[`forte_module`]`` to install all
            extra packages for using the forte module.
        component_name: the forte component that needs the module.

    """
    install_msg = (
        f" `{extra_module}` is not installed correctly."
        + f" Consider install {extra_module}"
    )
    pip_msg = f" via `pip install {extra_module}` "
    refer_msg = (
        f" or refer to extra requirement for {component_name}"
        + " at https://github.com/asyml/forte#installation"
        + f" for more information about installing {forte_module}. "
    )
    if pip_installable:
        error_msg = install_msg + pip_msg + refer_msg
    else:
        error_msg = install_msg + refer_msg
    return error_msg
