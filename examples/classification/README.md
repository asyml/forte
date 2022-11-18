## Prepare dataset
### Amazon Review Sentiment
amazon review sentiment(**ARS**) dataset is a binary classification dataset, and it has digit labels. `1` is
the negative and `2` is the positive. Each class has 1,800,000 training samples and 200,000 testing samples. 
The dataset can be downloaded from [link](https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz).


### banking77
Banking77 is a multi-class datasets. It has 77 classes which are fine-grained intents in a banking domain.
The train data can be downloaded from [link](https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv) and test data can be downloaded from [link](https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv).

## Run classifier
To simply test the script, one can use run cmd below directly to use sample data under the project root folder.
```
python examples/classification/amazon_review_sentiment.py
```

```
python examples/classification/bank_customer_intent.py
```


If User wants to run the script on the full dataset, User can download the dataset and set the dataset path correctly in the script.
Under the `forte/examples/classification/` folder, User can run the following command to run the classifier.
```bash
python amazon_review_sentiment.py
```

```bash
python bank_customer_intent.py
```


## Reader Configuration
`ClassificationDatasetReader` is designed to read table-like classification datasets and currently it only support `csv` file which is a common file format. To use the reader correctly, User needs to check the dataset and configure the reader correspondingly. To better explain this, we will use ARS dataset as an example throughout the explanation.
* User needs to check column names of the dataset. In the example dataset, we have column names [label, title, content]. First, we need know the first column is about data labels. Second, we know the second and third column can be input text. Therefore, we can set `forte_data_fields` to be `['label', 'ft.onto.base_ontology.Title', 'ft.onto.base_ontology.Body']` that each element matches column names from dataset. `label` is just a keyword that reader needs to identify the label. `'ft.onto.base_ontology.Title'` and `'ft.onto.base_ontology.Body'` are two forte data entries that stores input text in proper wrappers. In some cases that dataset might contain unnecessary columns that User doesn't want to use at all, User can set corresponding list elements in `forte_data_fields` to `None` so that the reader can skip processing them. 
* User also needs to check if how many classes in the dataset to configure `index2class` which is a dictionary mapping from zero-based indices to class names. In ARS dataset, User can simply set it to
    `{0: "negative", 1: "positive"}`. For dataset with many classes such as banking77, User can initialize `class_names` to store a list of class names and then set 
    `index2class` to `dict(enumerate(class_names))`.
* User needs to check the first line of dataset if they are column names which are not input data. If it's the case, User needs to set `skip_k_starting_lines` to `1` to skip the first line. Otherwise, `skip_k_starting_lines` defaults to `0` which means not skipping the first line. In special cases when User wants to skip multiple lines, User can just set `skip_k_starting_lines` to the number of lines they want to skip.
* In some cases, dataset labels are digits rather than text. User needs to set `digit_label` to `True`. Then User needs to check if the dataset label starting with `1`, if so, User needs to set `one_based_index_label` to True.
