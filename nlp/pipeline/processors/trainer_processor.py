import numpy as np
from typing import Dict, List, Optional, Any
from nlp.pipeline.processors.base_processor import BaseProcessor
from nlp.pipeline.data.data_pack import DataPack


class DummyTrainerExtractor(BaseProcessor):
    """
    A dummy relation extractor
    """

    def __init__(self) -> None:
        super().__init__()
        """
        Processors should define the fields which are required in the input data
        """
        self.required_entries: Dict[str, Optional[List]] = {
            "context": None,
            "Token": ["span", "text"],
            "EntityMention": ["span", "text", "ner_type"]
        }
        self.label: Dict[str, Optional[List]] = {
            "RelationshipLink": ["parent.span", "child.text", "rel_type"]
        }

    def process(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Defines the process step of the processor.

        Args:
            input_dict

        Returns:
            output_dict
        """

        # Check the existence of required entries and fields
        self.input_entry_existence_check(input_dict)

        contexts = input_dict["context"]
        tokens_span = input_dict["Token"]["span"]
        entities_span = input_dict["EntityMention"]["span"]
        if isinstance(contexts, str):
            contexts = [contexts]
            tokens_span = [tokens_span]
            entities_span = [entities_span]

        output_dict = {
            "RelationshipLink": {
                "parent.span": [],
                "child.span": [],
                "rel_type": [],
            }
        }

        for context, token, entity in zip(contexts, tokens_span, entities_span):
            parent = []
            child = []
            ner_type = []

            entity_num = len(entity)
            for i in range(entity_num):
                for j in range(i + 1, entity_num):
                    parent.append(entity[i])
                    child.append(entity[j])
                    ner_type.append("founded_by")

            output_dict["RelationshipLink"]["parent.span"].append(
                np.array(parent))
            output_dict["RelationshipLink"]["child.span"].append(
                np.array(child))
            output_dict["RelationshipLink"]["rel_type"].append(
                np.array(ner_type))

        return output_dict

    def input_entry_existence_check(self, inputs):
        for entry, fields in self.required_entries.items():
            if entry not in inputs.keys():
                raise KeyError(
                    f"Entry {entry} is required for "
                    f"{self.__class__.__name__} "
                    f"but missing in the inputs dict"
                )
            if fields is None:
                continue
            if not isinstance(inputs[entry], dict):
                raise KeyError(
                    f"Fields {fields} in {entry} are required for "
                    f"{self.__class__.__name__} "
                    f"but missing in the inputs dict"
                )
            for f in fields:
                if f not in inputs[entry].keys():
                    raise KeyError(
                        f"Field {f} in {entry} is required for "
                        f"{self.__class__.__name__} "
                        f"but missing in the inputs dict"
                    )

    def pack(self,
             output_dict: Dict,
             data_packs: List[DataPack],
             start_from: int = 0):        # Add corresponding fields to data_pack
        for result_key, result_value in output_dict.items():
            # Custom function of how to add the value back.
            pass
        pass

