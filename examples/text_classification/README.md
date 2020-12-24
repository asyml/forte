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
python imdb_format.py
```

The next step is to generate augment training data (using your favorite back translation model) and output to a TXT file. Each example in the file should correspond to the same line in `train.csv` (without headers).

For demonstration purpose, we provide the processed and augmented [data files](https://drive.google.com/file/d/1OKrbS76mbGCIz3FcFQ8-qPpMTQkQy8bP/view?usp=sharing). Place the CSV and txt files in directory `data/IMDB`.

### Train

To train the baseline model without UDA:

 ```bash
python main.py --do-train --do-eval --do-test
```

To train with UDA:

 ```bash
python main.py --do-train --do-eval --do-test --use-uda
```

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

You can further improve the performance by tuning hyperparameters, generate better back-translation data, using a larger BERT model, using a larger `max_seq_length` etc.

## Using the UDAIterator

Here is a brief tutorial to using Forte's `UDAIterator`. You can also refer to the `run_uda` function in `main.py`.

### Initialization

First, we initialize the `UDAIterator` with the supervised and unsupervised data:

```
iterator = tx.data.DataIterator(
    {"train": train_dataset, "eval": eval_dataset}
)

unsup_iterator = tx.data.DataIterator(
    {"unsup": unsup_dataset}
)

uda_iterator = UDAIterator(
    iterator,
    unsup_iterator,
    softmax_temperature=1.0,
    confidence_threshold=-1,
    reduction="mean")
```

The next step is to tell the iterator which dataset to use, and initialize the internal iterators:

```
uda_iterator.switch_to_dataset_unsup("unsup")
uda_iterator.switch_to_dataset("train", use_unsup=True)
uda_iterator = iter(uda_iterator) # call iter() to initialize the internal iterators
```

### Training with UDA

The UDA loss is the KL divergence between the the output probabilities of original input and augmented input. Here, we define `unsup_forward_fn` to calculate the probabilities:

```
def unsup_forward_fn(batch):
    input_ids = batch["input_ids"]
    segment_ids = batch["segment_ids"]
    input_length = (1 - (input_ids == 0).int()).sum(dim=1)

    aug_input_ids = batch["aug_input_ids"]
    aug_segment_ids = batch["aug_segment_ids"]
    aug_input_length = (1 - (aug_input_ids == 0).int()).sum(dim=1)

    logits, _ = model(input_ids, input_length, segment_ids)
    logits = logits.detach()  # gradient does not propagate back to original input
    aug_logits, _ = model(aug_input_ids, aug_input_length, aug_segment_ids)
    return logits, aug_logits
```

Then, `UDAIterator.calculate_uda_loss` computes the UDA loss for us. Inside the training loop, we compute the supervised loss as usual (or with a TSA schedule), and add the unsupervised loss to produce the final loss:

```
# ...
# Inside Training Loop:
# sup loss
logits, _ = model(input_ids, input_length, segment_ids)
loss = _compute_loss_tsa(logits, labels, scheduler.last_epoch,\
    num_train_steps)
# unsup loss
unsup_logits, unsup_aug_logits = unsup_forward_fn(unsup_batch)
unsup_loss = uda_iterator.calculate_uda_loss(unsup_logits, unsup_aug_logits)

loss = loss + unsup_loss # unsup coefficient = 1
loss.backward()
# ...
```

You can read more about the TSA schedule from the UDA paper.

### Evaluation

For evaluation, we simply switch to the eval dataset. In the `for` loop we only need the supervised batch:

```
uda_iterator.switch_to_dataset("eval", use_unsup=False)
for batch, _ in uda_iterator:
#   do evaluation ...
```
