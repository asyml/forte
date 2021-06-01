#!/usr/bin/env bash
set -ev

export PYTHONPATH=$PYTHONPATH:`pwd`
ONTOLOGY_SPECS_DIR=forte/ontology_specs

pip install --upgrade pip
pip install --progress-bar off .
export PYTHONPATH=`pwd`

for file in $(find ${ONTOLOGY_SPECS_DIR} -type f)
    do
        if [[ ${file##*/} == *.json ]]
        then
            printf "\n\nGenerating ontology for the spec ${file}...\n"
            generate_ontology create -i ${file} --no_dry_run
        fi
    done
