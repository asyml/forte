#!/usr/bin/env bash
set -ev

export PYTHONPATH=$PYTHONPATH:`pwd`
ONTOLOGY_SPECS_DIR=forte/ontology_specs

for file in $(find ${ONTOLOGY_SPECS_DIR} -type f)
    do
        if [[ ${file##*/} == *.json ]]
        then
            printf "\n\nGenerating ontology for the spec ${file}...\n"
            python scripts/generate_ontology/__main__.py create -i ${file} --no_dry_run
        fi
    done
