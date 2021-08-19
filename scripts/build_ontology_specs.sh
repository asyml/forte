#!/usr/bin/env bash
set -ev

export PYTHONPATH=$PYTHONPATH:$(pwd)
ONTOLOGY_SPECS_DIR=forte/ontology_specs

for file in $(find ${ONTOLOGY_SPECS_DIR} -type f); do
  if [[ ${file##*/} == *.json ]]; then
    printf "\n\nGenerating ontology for the spec ${file}...\n"
    python -m forte.command_line.generate_ontology create -i ${file} -a -r
  fi
done
