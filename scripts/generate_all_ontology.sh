#!/usr/bin/env bash
set -ev
ONTOLOGY_SPECS_DIR=ontology_specs

for file in $(find ${ONTOLOGY_SPECS_DIR} -type f)
    do
        if [[ ${file##*/} == *.json ]]
        then
            printf "\n\nGenerating ontology for the spec ${file}...\n"
            generate_ontology create -i ${file} --no_dry_run
        fi
    done