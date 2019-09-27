import torch

from texar.torch.data import RecordData
from texar.torch.modules import BERTEncoder
from texar.torch.data import BERTTokenizer

from processor import get_processor_class

max_seq_length = 512

print(f"Processing the IMDB reviews...")
processor_class = get_processor_class("IMDB")
imdb_processor = processor_class("../../aclImdb")
train_examples = imdb_processor.get_train_examples()
dev_examples = imdb_processor.get_dev_examples()
reviews = [example.text_a for dataset in [train_examples, dev_examples] for example in dataset]

# create a BERT tokenizer
vocab_file = "./pretrained_models/uncased_L-12_H-768_A-12/vocab.txt"
tokenizer = BERTTokenizer.load(vocab_file)

# BERT encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = BERTEncoder(pretrained_model_name="bert-base-uncased")
encoder.to(device)

print(f"Encoding the text using BERT Tokenizer...")
feature_original_types = {
        "id": ["int64", "FixedLenFeature"],
        "input_ids": ["int64", "FixedLenFeature", max_seq_length],
        "segment_ids": ["int64", "FixedLenFeature", max_seq_length],
        "text": ["str", "FixedLenFeature"]
    }

with RecordData.writer("./data/imdb2.pkl", feature_original_types) as writer:
    for idx, review in enumerate(reviews):
        review = review[:tokenizer.max_len]
        input_ids, segment_ids, _ = tokenizer.encode_text(text_a=review)
        feature = {
            "id": idx,
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "text": review
        }

        writer.write(feature)



