class OntologyGenerationWarning(UserWarning):
    pass


class DirectoryAlreadyPresentWarning(OntologyGenerationWarning):
    pass


class DuplicateEntriesWarning(OntologyGenerationWarning):
    pass


class DuplicatedAttributesWarning(OntologyGenerationWarning):
    pass


class OntologySpecError(ValueError):
    pass


class OntologySpecValidationError(OntologySpecError):
    pass


class OntologySourceNotFoundException(OntologySpecError):
    pass


class OntologyAlreadyGeneratedException(OntologySpecError):
    pass


class ParentEntryNotDeclaredException(OntologySpecError):
    pass


class ParentEntryNotSupportedException(OntologySpecError):
    pass


class TypeNotDeclaredException(OntologySpecError):
    pass


class NoDefaultClassAttributeException(OntologySpecError):
    pass


class UnsupportedTypeException(OntologySpecError):
    pass


class InvalidIdentifierException(ValueError):
    pass


class CodeGenerationException(BaseException):
    pass
