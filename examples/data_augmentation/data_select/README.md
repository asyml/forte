## Data Selector for Data Augmentation

The data selector is used for pre-select data from the training dataset that are suitable for data augmentation tasks.


### Examplge Usage

To create an elastic search indexer for a specific dataset, please refer to `data_select_index_pipeline.py` and `create_indexer_example.py` for details.

To build a random or query based selector processor that selects data from the indexer, please refer to `forte.processors.base.data_selector_for_da.py`. Refer to `data_select_and_augment_example.py` for how to insert this processor in a pipeline.


### Quick start

1. Install the `elasticsearch` backend [here](https://github.com/elastic/elasticsearch#installation)

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

