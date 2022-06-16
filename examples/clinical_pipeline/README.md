## A Clinical Information Processing Example

This project shows how we can construct projects to make Forte and Stave work
 side by side.
 
## Install extra dependencies

To install the latest code directly from source,

```bash
pip install git+https://git@github.com/asyml/forte-wrappers#egg=forte.elastic\&subdirectory=src/elastic
pip install git+https://git@github.com/asyml/forte-wrappers#egg=forte.spacy\&subdirectory=src/spacy
pip install git+https://git@github.com/asyml/forte-wrappers#egg=forte.spacy\&subdirectory=src/nltk
```

To install from PyPI,
```bash
pip install forte.elastic
pip install forte.spacy
pip install forte.nltk
```

## Downloading the models

In this example, we use an NER model fine-tuned on the NCBI-disease dataset, which predicts 
named entities corresponding to diseases, symptoms and related concepts. 
Before running the pipeline, we need to download the models

```bash
python ./download_models.py 
```

**Note**: The above script will save the model in `resources/NCBI-disease`. Use `--path` option to save the 
model into a different directory.

## Prepare elastic searcher
You also need to set up elastic searcher by following guide below to run the pipeline.

https://www.elastic.co/guide/en/elasticsearch/reference/current/starting-elasticsearch.html


## Run indexer and Stave
First, you should start an Elastic Indexer backend.

Then, to start the Stave server that our pipeline will connect to for visualization purposes, run
```bash
stave -s start -o -l -n 8899
```

## Run demo pipeline

Now, open a new terminal, other than the one running stave server. You can run the following command to parse some files and index them.
```bash
python clinical_processing_pipeline.py /path/to/mimiciii/1.4/NOTEEVENTS.csv.gz /path_to_sample_output 100 1
```

The last argument, `run_ner_pipeline` is whether we wish to run the NER pipeline or if we just need the remote pipeline connection to Stave. We set it to `1` if we want to run the NER pipeline and setup a connection with Stave, else `0` for just the connection.
 
Hence, if you just wish to run the demo pipeline with existing database entries, and wish to just connect with Stave for visualization, you can run this command:

```bash
python clinical_processing_pipeline.py ./ ./ 100 0
```

Here, we also write out the raw data pack to `/path_to_sample_output`, and only
index the first 100 notes. Remove the `100` parameter to index all documents.

## Visualization

You can go ahead and open `http://localhost:8899` on your browser to access Stave UI.
Next, you will see 2 projects, named as `clinical_pipeline_base` and `clinical_pipeline_chat` by default.

Click on `clinical_pipeline_chat` and then the document that resides within to go to the chatbot/search UI. Enter the keywords you want to search for in the elasticsearch indices. The pipeline would then return a bunch of documents that match your keywords. Click on those document links to access the Annotation Viewer UI for those documents.
