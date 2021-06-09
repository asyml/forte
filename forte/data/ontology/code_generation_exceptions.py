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
    Exception classes for ontology code generation.
"""


class OntologyGenerationWarning(UserWarning):
    """General warning for ontology generation."""

    pass


class DirectoryAlreadyPresentWarning(OntologyGenerationWarning):
    """The directory that the code will be written to already exists."""

    pass


class DuplicateEntriesWarning(OntologyGenerationWarning):
    """The entry is defined multiple times."""

    pass


class DuplicatedAttributesWarning(OntologyGenerationWarning):
    """The attribute is defined multiple times."""

    pass


class OntologySpecError(ValueError):
    """General error related to ontology specification."""

    pass


class OntologySpecValidationError(OntologySpecError):
    """Error thrown during validating the ontology specification."""

    pass


class OntologySourceNotFoundException(OntologySpecError):
    """Raise when the import source specification cannot be found."""

    pass


class OntologyAlreadyGeneratedException(OntologySpecError):
    """Raise when generating the same ontology again, which is considered
    to be a cyclic or duplicated declaration."""

    pass


class ParentEntryNotDeclaredException(OntologySpecError):
    """Raise when the entry's parent entry is not previously defined."""

    pass


class ParentEntryNotSupportedException(OntologySpecError):
    """Raise when the entry uses a not allowed parent entry."""

    pass


class TypeNotDeclaredException(OntologySpecError):
    """
    Raise when the entry uses an attribute which is previously not defined.
    """

    pass


class UnsupportedTypeException(OntologySpecError):
    """Raise when the ontology generator do not support such type structure."""

    pass


class InvalidIdentifierException(OntologySpecValidationError):
    """Raise when the ontology contains illegal identifier."""

    pass


class CodeGenerationException(BaseException):
    """General errors during ontology generation."""

    pass
