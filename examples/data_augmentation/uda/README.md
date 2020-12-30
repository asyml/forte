## Unsupervised Data Augmentation for Text Classification

Unsupervised Data Augmentation or UDA is a semi-supervised learning method which achieves state-of-the-art results on a wide variety of language and vision tasks. For details, please refer to the [paper](https://arxiv.org/abs/1904.12848) and the [official repository](https://github.com/google-research/uda).

In this example, we demonstrate Forte's implementation of UDA using a simple BERT-based text classifier.

## Quick Start

### Install the dependencies

You need to install [texar-pytorch](https://github.com/asyml/texar-pytorch) first.

You will also need to install `tensor2tensor` if you want to perform back translation on your own data. We will cover this later.

### Get the IMDB data and back-translation models

We use the IMDB Text Classification dataset for this example. Use the following script to download the supervised and unsupervised training data to `data/IMDB_raw`. It will also download the pre-trained translation models for back-translation to the directory `back_trans`.

 ```bash
python download.py
```

### Preprocess

You can use the following script to preprocess the data.

 ```bash
python utils/imdb_format.py
```

This script does two things. It reads the raw data in TXT format and output two files `train.csv` and `test.csv`. It also splits the training set into sentences for back-translation. This is because the back-translation models are trained on sentences instead of long paragraphs.

### Generate back-translation data

**Notice:** back-translation is the most time-consuming step. If you just want to see the results, you can skip to the next section. Translating the whole dataset to French takes ~2 days on a GTX 1080 Ti. It takes another 2 days to translate back to English.

If you would like to play with the back-translation parameters or work with your own data, you need to generate back-translation data yourself. Here we provide an example of back-translation on the IMDB dataset.

First, you need to install `tensor2tensor` with Tensorflow 1.13. We provide a `requirements.txt` with the correct versions of dependencies. To install:

```
cd back_trans/
pip install -r requirements.txt
pip install --no-deps tensor2tensor==1.13
```

Then run the following command to run the back-translation (adapted from the original [UDA repo](https://github.com/google-research/uda/blob/master/back_translate/run.sh)):

```
cd back_trans/

# forward translation
t2t-decoder \
  --problem=translate_enfr_wmt32k \
  --model=transformer \
  --hparams_set=transformer_big \
  --hparams="sampling_method=random,sampling_temp=0.8" \
  --decode_hparams="beam_size=1,batch_size=16" \
  --checkpoint_path=checkpoints/enfr/model.ckpt-500000 \
  --output_dir=/tmp/t2t \
  --decode_from_file=train_split_sent.txt \
  --decode_to_file=forward_gen.txt \
  --data_dir=checkpoints

# backward translation
t2t-decoder \
  --problem=translate_enfr_wmt32k_rev \
  --model=transformer \
  --hparams_set=transformer_big \
  --hparams="sampling_method=random,sampling_temp=0.8" \
  --decode_hparams="beam_size=1,batch_size=16,alpha=0" \
  --checkpoint_path=checkpoints/fren/model.ckpt-500000 \
  --output_dir=/tmp/t2t \
  --decode_from_file=forward_gen.txt \
  --decode_to_file=backward_gen.txt \
  --data_dir=checkpoints

# merge sentences back to paragraphs
python merge_back_trans_sentences.py \
--input_file=backward_gen.txt \
--output_file=back_translate.txt \
--doc_len_file=train_doc_len.json
```

You can tune the `sampling_temp` parameter. See [here](https://github.com/google-research/uda#guidelines-for-hyperparameters) for more details.

The final result of the above commands is `back_translate.txt`. Each line in the file is a back translated example corresponding to the same line in `train.csv` (without the header).

Next, copy `back_translate.txt` to `data/IMDB/`.

```
cp back_translate.txt ../data/IMDB/
```

Of course, you can use a different name for the back translation file. Look at `config_data.py` to configure.

### Download preprocessed and augmented data

For demonstration purpose, we provide the processed and augmented data files: [download link](https://drive.google.com/file/d/1OKrbS76mbGCIz3FcFQ8-qPpMTQkQy8bP/view?usp=sharing). Place the CSV and txt files in directory `data/IMDB`.

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
