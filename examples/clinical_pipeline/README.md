## A Clinical Information Processing Example

This project shows how we can construct projects to make Forte and Stave work
 side by side.
 
## Install extra dependencies

In command line, we run

```bash
pip install git+https://git@github.com/asyml/forte-wrappers#egg=forte-wrappers[elastic,nltk]
```

## Downloading the models

In this example, we use an NER model fine-tuned on the NCBI-disease dataset, which predicts 
named entities corresponding to diseases, symptoms and related concepts. 
Before running the pipeline, we need to download the models

```bash
python ../biobert_ner/download_models.py 
```

**Note**: The above script will save the model in `resources/NCBI-disease`. Use `--path` option to save the 
model into a different directory.

## Run indexer
First, you should start an Elastic Indexer backend.

Second, you can run the following command to parse some files and index them.
```bash
python clinical_processing_pipeline.py /path/to/mimiciii/1.4/NOTEEVENTS.csv.gz /path_to_sample_output 10000
```

Here, we also write out the raw data pack to `/path_to_sample_output`, and only
index the first 10k notes. Remove the `10000` parameter to index all documents.

After the indexing is done, we are ready with the data processing part. Let's start the GUI.

## Stave 
First, set up Stave following the instructions.

Second, create an empty project with the [default ontology](https://github.com/asyml/forte/blob/master/forte/ontology_specs/base_ontology.json),
 now record the project id.

Set up the following environment variables:
```bash
export stave_db_path=[path_to_stave]/simple_backend/db.sqlite3
export url_stub=http://localhost:3000
export query_result_project_id=[the project id above]
```

Now, create another project with default ontology.

Upload the `query_chatbot.json` file (you can find it in the directory of the README) to the project.
