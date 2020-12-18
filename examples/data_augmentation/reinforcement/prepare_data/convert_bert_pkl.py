import argparse
import logging
import texar.torch as tx

from examples.data_augmentation.reinforcement.utils import data_utils
from examples.data_augmentation.reinforcement.config import config_data


parser = argparse.ArgumentParser()
parser.add_argument(
    '--pretrained-model-name', type=str, default='bert-base-uncased',
    choices=tx.modules.BERTEncoder.available_checkpoints(),
    help="Name of the pre-trained downstream checkpoint to load.")
args = parser.parse_args()


def main():
    """Prepares data.
    """
    logging.info("Loading data")

    csv_data_dir = config_data.pickle_data_dir
    output_dir = config_data.pickle_data_dir
    tx.utils.maybe_create_dir(output_dir)

    processor = data_utils.IMDbProcessor()

    num_classes = len(processor.get_labels())
    num_train_data = len(processor.get_train_examples(csv_data_dir))
    logging.info(
        'num_classes:%d; num_train_data:%d', num_classes, num_train_data)

    tokenizer = tx.data.BERTTokenizer(
        pretrained_model_name=args.pretrained_model_name)

    data_utils.prepare_record_data(
        processor=processor,
        tokenizer=tokenizer,
        data_dir=csv_data_dir,
        max_seq_length=config_data.max_seq_length,
        output_dir=output_dir,
        feature_types=config_data.feature_types)


if __name__ == "__main__":
    main()
