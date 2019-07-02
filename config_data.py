import re

train_path = "ner_data/conll03_english/train/"
val_path = "ner_data/conll03_english/dev/"
test_path = "ner_data/conll03_english/test"

alphabet_directory = "ner_data/alphabets/ner_crf/"
num_epochs = 200
batch_size_tokens = 512
test_batch_size = 16

max_char_length = 45
num_char_pad = 2
normalize_digit = True
digit_re = re.compile(r"\d")