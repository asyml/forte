"""
    Utility functions for ontology generation.
"""
import json
import jsonschema


def validate_json_schema(input_filepath: str, validation_filepath: str):
    """
    Validates the input json schema using validation meta-schema provided in
    `validation_filepath` according to the specification in
    `http://json-schema.org`.
    If the tested json is not valid, a `jsonschema.exceptions.ValidationError`
    is thrown.
    Args:
        input_filepath: Filepath of the json schema to be validated
        validation_filepath: Filepath of the valiodation specification
    """
    with open(validation_filepath, 'r') as validation_json_file:
        validation_schema = json.loads(validation_json_file.read())
    with open(input_filepath, 'r') as input_json_file:
        input_schema = json.loads(input_json_file.read())
    jsonschema.Draft6Validator(validation_schema).validate(input_schema)
