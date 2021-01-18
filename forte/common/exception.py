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
Forte defined exceptions.
"""

__all__ = [
    "PackIndexError",
    "IncompleteEntryError",
    "EntryNotFoundError",
    "ProcessorConfigError",
    "PackDataException",
    "ProcessFlowException",
    "ProcessExecutionException",
    "ValidationError",

]


class PackIndexError(Exception):
    r"""Raise this error when there is a problem accessing the indexes pack
    data.
    """
    pass


class IncompleteEntryError(Exception):
    r"""Raise this error when the entry is not complete.
    """
    pass


class EntryNotFoundError(ValueError):
    r"""Raise this error when the entry is not found in the data pack.
    """
    pass


class ProcessorConfigError(ValueError):
    r"""Raise this error when the there is a problem with the processor
    configuration.
    """
    pass


class PackDataException(Exception):
    r"""Raise this error when the data in pack is wrong."""
    pass


class ProcessFlowException(Exception):
    r"""Raise this when errors happen in flow control """
    pass


class ProcessExecutionException(Exception):
    r"""Raise this when process execution fail"""
    pass


class ValidationError(Exception):
    r"""Raise this error when input validate fail"""
    pass
