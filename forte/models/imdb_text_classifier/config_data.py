pickle_data_dir = "data/IMDB"
max_seq_length = 128
num_classes = 2
num_train_data = 24  # supervised data limit. max 25000

train_batch_size = 24
max_train_epoch = 3000
display_steps = 50  # Print training loss every display_steps; -1 to disable

eval_steps = 100  # Eval on the dev set every eval_steps; if -1 will eval every epoch
# Proportion of training to perform linear learning rate warmup for.
# E.g., 0.1 = 10% of training.
warmup_proportion = 0.1
eval_batch_size = 8
test_batch_size = 8

feature_types = {
    # Reading features from pickled data file.
    # E.g., Reading feature "input_ids" as dtype `int64`;
    # "FixedLenFeature" indicates its length is fixed for all data instances;
    # and the sequence length is limited by `max_seq_length`.
    "input_ids": ["int64", "stacked_tensor", max_seq_length],
    "input_mask": ["int64", "stacked_tensor", max_seq_length],
    "segment_ids": ["int64", "stacked_tensor", max_seq_length],
    "label_ids": ["int64", "stacked_tensor"]
}

train_hparam = {
    "allow_smaller_final_batch": False,
    "batch_size": train_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_types": feature_types,
        "files": "{}/train.pkl".format(pickle_data_dir)
    },
    "shuffle": True,
    "shuffle_buffer_size": None
}

eval_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": eval_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_types": feature_types,
        "files": "{}/eval.pkl".format(pickle_data_dir)
    },
    "shuffle": False
}

test_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": test_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_types": feature_types,
        "files": "{}/predict.pkl".format(pickle_data_dir)
    },
    "shuffle": False
}

# UDA config
tsa = True
tsa_schedule = "linear_schedule" # linear_schedule, exp_schedule, log_schedule

unsup_feature_types = {
    "input_ids": ["int64", "stacked_tensor", max_seq_length],
    "input_mask": ["int64", "stacked_tensor", max_seq_length],
    "segment_ids": ["int64", "stacked_tensor", max_seq_length],
    "label_ids": ["int64", "stacked_tensor"],
    "aug_input_ids": ["int64", "stacked_tensor", max_seq_length],
    "aug_input_mask": ["int64", "stacked_tensor", max_seq_length],
    "aug_segment_ids": ["int64", "stacked_tensor", max_seq_length],
    "aug_label_ids": ["int64", "stacked_tensor"]
}

unsup_hparam = {
    "allow_smaller_final_batch": True,
    "batch_size": train_batch_size,
    "dataset": {
        "data_name": "data",
        "feature_types": unsup_feature_types,
        "files": "{}/unsup.pkl".format(pickle_data_dir)
    },
    "shuffle": True,
    "shuffle_buffer_size": None,
}
