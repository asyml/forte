# A Complex Reader Example

Sometimes the dataset is quite complex to read. Here, we provide a real example
where we use Forte to parse Wikipedia articles using Datasets from DBpedia.

## Prerequisite
This script will use `rdflib` to run parse the input, you can install this by
installing Forte with the extra dependency:
```bash
pip install "forte[wikipedia]"
```

or simply install the rdf dependency in an existing Forte environment:
```bash
pip install rdflib==4.2.2
```

# Running the script
```bash
python wiki_dump_parse.py [dbpedia datasets directory] [output directory]
```

There are quite a few datasets needed to be downloaded into the input folder. 
You can download them from [DBpedia](https://downloads.dbpedia.org/current/core-i18n/en/). 

Before trying out with the full dataset, you can try this script with the sample data located at 
https://github.com/asyml/forte/tree/master/data_samples/dbpedia:

```bash
python wiki_dump_parse.py ../../data_samples/dbpedia sample_output True
```

You should be able to find the parsed results in the output (sample_output) folder. There are
multiple steps in the process, and the intermediate results are written to corresponding sub-folders.

When you are ready to try out the full example, we provide a simple script to download the full 
datasets, note that these datasets are sizable:

```bash
# Will download data to the a directory named "dbpedia".
./data_download.sh
```

And you can now try parse the real DBpedia dataset
```bash
# You should find the processed data packs in "output".
python wiki_dump_parse.py dbpedia output
```