"""
chatbot config
"""

max_seq_length = 128
pickle_data_dir = "data/"
train_batch_size = 16
eval_batch_size = 128
test_batch_size = 128
num_train_data = 131424
max_train_epoch = 1
warmup_proportion = 0.1

display_steps = 50
eval_steps = 500  # Eval on the dev set every eval_steps; -1 to disable

feature_types = {
    # Reading features from pickled data file.
    "sent_a_input_ids": ["int64", "stacked_tensor", max_seq_length],
    "sent_a_seq_len": ["int64", "stacked_tensor"],
    "sent_a_segment_ids": ["int64", "stacked_tensor", max_seq_length],
    "sentence_a": ["str", "stacked_tensor"],
    "sent_b_input_ids": ["int64", "stacked_tensor", max_seq_length],
    "sent_b_seq_len": ["int64", "stacked_tensor"],
    "sent_b_segment_ids": ["int64", "stacked_tensor", max_seq_length],
    "sentence_b": ["str", "stacked_tensor"],
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
    "shuffle_buffer_size": 100
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
        "files": "{}/test.pkl".format(pickle_data_dir)
    },
    "shuffle": False
}
