#!/usr/bin/env bash
# If the ontology code output are changed, can run this script to regenerate the expected test cases.
generate_ontology create -i tests/forte/data/ontology/test_specs/example_ontology.json -r -o tests/forte/data/ontology/test_outputs
generate_ontology create -i tests/forte/data/ontology/test_specs/example_complex_ontology.json -r -o tests/forte/data/ontology/test_outputs
generate_ontology create -i tests/forte/data/ontology/test_specs/example_multi_module_ontology.json -r -o tests/forte/data/ontology/test_outputs
generate_ontology create -i tests/forte/data/ontology/test_specs/race_qa_onto.json -r -o tests/forte/data/ontology/test_outputs