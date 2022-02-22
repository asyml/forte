import functools
import importlib

__all__ = ["EntryTypeGenerator"]


def _get_type_attributes():
    # TODO: implement get type_attributes using entry class hierachy
    type_attributes = {}

    return type_attributes


def _get_entry_attribute_by_class(input_entry_class: str):
    modname, _, clsname = input_entry_class.rpartition(".")
    mod = importlib.import_module(modname)
    try:
        return list(getattr(mod, clsname).__annotations__.keys())
    except AttributeError:
        return []


class EntryTypeGenerator:
    def __init__(self):
        pass

    @staticmethod
    @functools.lru_cache(1)
    def get_type_attributes():
        """
        While initializing the new data structure, we need a list of all the valid annotations,
        links and groups. For example, for annotations, we want to have a list of types as strings,
        ["Token", "Subword", "Sentence", ...], including all the types defined in ft.onto and
        ftx.onto. Please notice that users could also create their own types of data which inherit
        "forte.data.ontology.core.Entry".
        For each type, we want to obtain all the attributes. For example, for sentence we want to
        get a list of ["speaker", "part_id", "sentiment", "classification", "classifications"]. We
        want to get the attributes for each type of entry as a dictionary. For example:

        .. code-block:: python

            type_attributes = {
                "Token": ["pos", "ud_xpos", "lemma", "chunk", "ner", "sense",
                        "is_root", "ud_features", "ud_misc"],
                "Document": ["document_class", "sentiment", "classifications"],
                "Sentence": ["speaker", "part_id", "sentiment",
                            "classification", "classifications"],
            }

        """
        return _get_type_attributes()

    @staticmethod
    def get_entry_attribute_by_class(input_entry_class: str):
        """
        For each type, we want to obtain all the attributes. For example, for sentence we want to
        get a list of ["speaker", "part_id", "sentiment", "classification", "classifications"].
        We want to get the attributes for each type of entry as a dictionary. The solution
        looks like the following:

        .. code-block::python

            # input can be a string
            entry_name = "ft.onto.base_ontology.Sentence"
            # or a class
            entry_class = ft.onto.base_ontology.Sentence

            # function signature
            def get_entry_attribute_by_class(input_entry_class: Union[str, class]):

            # output
            ["speaker", "part_id", "sentiment", "classification", "classifications"]

        """
        return _get_entry_attribute_by_class(input_entry_class)
