## Data Selector for Data Augmentation

The data selector is used for pre-select data from the training dataset that are suitable for data augmentation tasks. Here, we show an example by using an elastic search backend to select data randomly or based on user queries.


### Examplge Usage

##### To create an elastic search indexer for a specific dataset: 

```python
from examples.data_augmentation.data_select import CreateIndexerPipeline

# Build a reader to read data into datapacks.
reader = SomeReader()
# Set indexer config.
# fields.content: texts to index from.
# fields.pack_info: stores the serialized datapack. 
indexer_config = {
    "batch_size": 5,
    "fields":
        ["doc_id", "content", "pack_info"],
    "indexer": {
        "name": "ElasticSearchIndexer",
        "hparams":
            {"index_name": index_name,
             "hosts": "localhost:9200",
             "algorithm": "bm25"},
        "other_kwargs": {
            "request_timeout": 10,
            "refresh": False
        }
    }
}
# Build the indexer creator pipeline that stores the datapacks into elastic search backend.
nlp = CreateIndexerPipeline(reader=reader, reader_config=reader_config, indexer_config=indexer_config)
# Start document indexing from data_dir.
nlp.create_index(data_dir)
```

Please refer to `data_select_index_pipeline.py` and `create_indexer_example.py` for details.

##### To select data from the indexer:

```python
# Random data select.
from forte.processors.base.data_selector_for_da import RandomDataSelector

data_selector_config = {"index_config": 
                            {"index_name": indexer_name}, 
                        "size": size}

nlp: Pipeline[DataPack] = Pipeline()
nlp.set_reader(RandomDataSelector(), config=data_selector_config)
nlp.initialize()

for idx, pack in nlp.process_dataset():
	# Do something.

# Query-based data selector.
from forte.processors.base.data_selector_for_da import QueryDataSelector

data_selector_config = {"index_config":
                            {"index_name": index_name},
                        "size": size,
                        "field": "content"}	# field: the field to index from

nlp: Pipeline[DataPack] = Pipeline()
nlp.set_reader(QueryDataSelector(), config=data_selector_config)
nlp.initialize()

# file_path: the file to store queries.
# Format of file: one query per line.
# 	query 1
# 	query 2
# 	...
for pack in nlp.process_dataset(file_path):
	# Do something.
```
Please refer to `forte.processors.base.data_selector_for_da.py` for details of the data selector processor. Refer to `data_select_and_augment_example.py` for details of how to insert the data selector processor into a pipeline.


### Quick start

1. Install the `elasticsearch` backend [here](https://github.com/elastic/elasticsearch#installation).

2. Start the `elasticsearch` backend:
```bash
bin/elasticsearch
```

In `examples.data_augmentation.data_select` directory:

3. Create an elastic search indexer with input data:
```bash
python create_indexer_example.py
```

4. Search from the same indexer, and perform data augmentation:
```bash
python data_select_and_augment_example.py
```
