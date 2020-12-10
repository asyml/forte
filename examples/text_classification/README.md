## Unsupervised Data Augmentation for Text Classification

Unsupervised Data Augmentation or UDA is a semi-supervised learning method which achieves state-of-the-art results on a wide variety of language and vision tasks. For details, please refer to the [paper](https://arxiv.org/abs/1904.12848) and the [official repository](https://github.com/google-research/uda).

In this example, we demonstrate Forte's implementation of UDA using a simple BERT-based text classifier.

## Quick Start

### Install the dependencies

You need to install [texar-pytorch](https://github.com/asyml/texar-pytorch) first.

### Get the IMDB data

We use the IMDB Text Classification dataset for this example. Use the following script to download the supervised and unsupervised training data.

 ```bash
python download_imdb.py
```

### Preproces and generate augmented data

You can use the following script to process the data into CSV format.

 ```bash
python utils/imdb_format.py --raw_data_dir=data/IMDB_raw/aclImdb --train_id_path=data/IMDB_raw/train_id_list.txt --output_dir=data/IMDB
```

The next step is to generate augment training data (using your favorite back translation model) and output to a TXT file. Each example in the file should correspond to the same line in `train.csv`.

For demonstration purpose, we provide the processed and augmented [data files](https://drive.google.com/file/d/1OKrbS76mbGCIz3FcFQ8-qPpMTQkQy8bP/view?usp=sharing). Place the CSV and txt files in directory `data/IMDB`.

### Train

 ```bash
python main.py
```

To train the baseline model without UDA, use `model.run()` instead of `model.run_uda()`.

To change the hyperparameters, please see `config_data.py`. You can also change the number of labeled examples used for training (`num_train_data`).

#### GPU Memory Issue:

According to the authors' [guideline for hyperparameters](https://github.com/google-research/uda#general-guidelines-for-setting-hyperparameters), longer sequence length and larger batch size lead to better performances. The sequence length and batch size are limited by the GPU memory. By default, we use `max_seq_length=128` and `batch_size=24` to run on a GTX1080Ti with 11GB memory.

## Results

With the provided data, you should be able to achieve performance similar to the following:

| Number of Labeled Examples | BERT Accuracy | BERT+UDA Accuracy|
| -------------------------- | ------------- | ------------------ |
| 24                         | 61.54         | 84.92              |
| 25000                      | 89.68         | 90.19              |

When training with 24 examples, we use the Training Signal Annealing technique which can be turned on by setting `tsa=True`.

You can further improve the performance by tuning hyperparameters, generate better back-translation data, using a larger BERT model, etc.
