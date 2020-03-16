#!/usr/bin/env bash
set -ev
ONTOLOGY_SPECS_DIR=ontology_specs

pip install --upgrade pip
pip install --progress-bar off .
export PYTHONPATH=`pwd`

for file in $(find ${ONTOLOGY_SPECS_DIR} -type f)
    do
        if [[ ${file##*/} == *.json ]]
        then
            printf "\n\nGenerating ontology for the spec ${file}...\n"
            python scripts/generate_ontology/__main__.py create -i ${file} --no_dry_run
        fi
    done
