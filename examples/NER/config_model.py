import torch.nn as nn

word_embed_dim = 100
char_embed_dim = 30
rnn_hidden_size = 128
output_hidden_size = 128
dropout_rate = 0.3

load_glove = True

word_emb = {
    "dim": word_embed_dim,
    # TODO(haoransh): currently we leave the dropout out of this definition
    #  scope to make code more consistent with character encoding
    # "dropout_rate": 0.3,
    # "dropout_strategy": 'item'
}
embedding_path = "ner_data/glove/glove.6B/glove.6B.100d.txt"

char_emb = {
    "dim": char_embed_dim,
    'initializer': {
        'type': 'normal_'
    },
}


# TODO(haoransh): we may consider wrap this in Texar modules later
char_cnn_conv = {
    "in_channels": char_embed_dim,
    "out_channels": 30,
    "kernel_size": 3,
    "padding": 2,
}

bilstm_sentence_encoder = {
    "rnn_cell_fw": {
        'kwargs': {
            'num_units': rnn_hidden_size,
        },
    },
    "rnn_cell_share_config": True,

    "output_layer_fw": {
        "num_layers": 0,
    },
    "output_layer_share_config": True,
}


learning_rate = 0.01
momentum = 0.9
decay_interval = 1
decay_rate = 0.05

random_seed = 1234

# opt = {
#     "optimizer": {
#         "type": "MomentumOptimizer",
#         "kwargs": {"learning_rate": learning_rate,
#                    "momentum": 0.9,
#                    "use_nesterov": True}
#     },
#     "learning_rate_decay": {
#         "type": "inverse_time_decay",
#         "kwargs": {
#             "decay_steps": decay_interval,
#             "decay_rate": decay_rate,
#             "staircase": True
#         },
#         "start_decay_step": 1
#     }
# }

initializer = nn.init.xavier_uniform_

# "The path to save model.",
model_path = "best_ner_crf_model.ckpt"
