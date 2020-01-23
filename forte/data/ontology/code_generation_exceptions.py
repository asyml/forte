class OntologyGenerationWarning(UserWarning):
    pass


class DirectoryAlreadyPresentWarning(OntologyGenerationWarning):
    pass


class DuplicateEntriesWarning(OntologyGenerationWarning):
    pass


class OntologySpecError(ValueError):
    pass


class ImportOntologyNotFoundException(OntologySpecError):
    pass


class ImportOntologyAlreadyGeneratedException(OntologySpecError):
    pass


class ParentEntryNotDeclaredException(OntologySpecError):
    pass


class TypeNotDeclaredException(OntologySpecError):
    pass


class NoDefaultClassAttributeException(OntologySpecError):
    pass


class UnsupportedTypeException(OntologySpecError):
    pass


class InvalidIdentifierException(ValueError):
    pass


class DuplicatedAttributesWarning(OntologyGenerationWarning):
    pass