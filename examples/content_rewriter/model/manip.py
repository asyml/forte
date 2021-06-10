"""
Textontent Manipulation
3-gated copy net.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os

import numpy as np
import tensorflow as tf
import texar as tx
from tensorflow.contrib.seq2seq.python.ops.beam_search_decoder import tile_batch
from texar.core import get_train_op

from copy_net import CopyNetWrapper
from utils_e2e_clean import (
    x_strs,
    x_fields,
    y_strs,
    get_scope_name_of_train_op,
    get_scope_name_of_summary_op,
    corpus_bleu,
)


# pylint: disable=invalid-name, no-member, too-many-locals, global-statement
# pylint: disable=undefined-loop-variable, unused-variable, chained-comparison
# pylint: disable=unexpected-keyword-arg, no-value-for-parameter,
# pylint: disable=protected-access, unused-argument, global-variable-undefined
# pylint: disable=attribute-defined-outside-init


class Config:
    copy_x = True
    copy_y_ = False
    attn_x = True
    attn_y_ = True
    expr_name = "model/e2e_model/demo"
    disabled_vocab_size = 0
    eps = 1e-10
    add_bleu_weight = False
    exact_cover_w = 2.5

    coverage = True
    restore_from = ""
    rec_w = 0.8

    dir_summary = os.path.join(expr_name, "log")
    dir_model = os.path.join(expr_name, "ckpt")
    dir_best = os.path.join(expr_name, "ckpt-best")
    ckpt_model = os.path.join(dir_model, "model.ckpt")
    ckpt_best = os.path.join(dir_best, "model.ckpt")

    copy_flag = copy_x or copy_y_
    attn_flag = attn_x or attn_y_

    config_model = importlib.import_module(
        "examples.content_rewriter.model.config_model_clean"
    )
    config_train = importlib.import_module(
        "examples.content_rewriter.model.config_train"
    )
    config_data = importlib.import_module(
        "examples.content_rewriter.model.config_data_e2e_clean"
    )

    @classmethod
    def set_path(cls, path: str):
        cls.expr_name = path
        cls.dir_summary = os.path.join(cls.expr_name, "log")
        cls.dir_model = os.path.join(cls.expr_name, "ckpt")
        cls.dir_best = os.path.join(cls.expr_name, "ckpt-best")
        cls.ckpt_model = os.path.join(cls.dir_model, "model.ckpt")
        cls.ckpt_best = os.path.join(cls.dir_best, "model.ckpt")


def get_optimistic_restore_variables(ckpt_path, graph=tf.get_default_graph()):
    reader = tf.train.NewCheckpointReader(ckpt_path)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted(
        [
            (var.name, var.name.split(":")[0])
            for var in tf.global_variables()
            if var.name.split(":")[0] in saved_shapes
        ]
    )
    restore_vars = []
    for var_name, saved_var_name in var_names:
        var = graph.get_tensor_by_name(var_name)
        var_shape = var.get_shape().as_list()
        if var_shape == saved_shapes[saved_var_name]:
            restore_vars.append(var)
    return restore_vars


def get_optimistic_saver(ckpt_path, graph=tf.get_default_graph()):
    return tf.train.Saver(
        get_optimistic_restore_variables(ckpt_path, graph=graph)
    )


def build_model(data_batch, data, step):
    batch_size, num_steps = [
        tf.shape(data_batch["x_value_text_ids"])[d] for d in range(2)
    ]
    vocab = data.vocab("y_aux")

    id2str = "<{}>".format
    bos_str, eos_str = map(id2str, (vocab.bos_token_id, vocab.eos_token_id))

    def single_bleu(ref, hypo):
        ref = [id2str(u if u != vocab.unk_token_id else -1) for u in ref]
        hypo = [id2str(u) for u in hypo]

        ref = tx.utils.strip_special_tokens(
            " ".join(ref), strip_bos=bos_str, strip_eos=eos_str
        )
        hypo = tx.utils.strip_special_tokens(" ".join(hypo), strip_eos=eos_str)

        return 0.01 * tx.evals.sentence_bleu(references=[ref], hypothesis=hypo)

    def batch_bleu(refs, hypos):
        return np.array(
            [single_bleu(ref, hypo) for ref, hypo in zip(refs, hypos)],
            dtype=np.float32,
        )

    def lambda_anneal(step_stage):

        print("==========step_stage is {}".format(step_stage))
        if step_stage <= 1:
            rec_weight = 1
        elif step_stage > 1 and step_stage < 2:
            rec_weight = Config.rec_w - step_stage * 0.1
        return np.array(rec_weight, dtype=tf.float32)

    # losses
    losses = {}

    # embedders
    embedders = {
        name: tx.modules.WordEmbedder(
            vocab_size=data.vocab(name).size, hparams=hparams
        )
        for name, hparams in Config.config_model.embedders.items()
    }

    # encoders
    y_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=Config.config_model.y_encoder
    )
    x_encoder = tx.modules.BidirectionalRNNEncoder(
        hparams=Config.config_model.x_encoder
    )

    def concat_encoder_outputs(outputs):
        return tf.concat(outputs, -1)

    def encode(ref_flag):
        y_str = y_strs[ref_flag]
        sent_ids = data_batch["{}_text_ids".format(y_str)]
        sent_embeds = embedders["y_aux"](sent_ids)
        sent_sequence_length = data_batch["{}_length".format(y_str)]
        sent_enc_outputs, _ = y_encoder(
            sent_embeds, sequence_length=sent_sequence_length
        )
        sent_enc_outputs = concat_encoder_outputs(sent_enc_outputs)

        x_str = x_strs[ref_flag]
        sd_ids = {
            field: data_batch["{}_{}_text_ids".format(x_str, field)][:, 1:-1]
            for field in x_fields
        }
        sd_embeds = tf.concat(
            [
                embedders["x_{}".format(field)](sd_ids[field])
                for field in x_fields
            ],
            axis=-1,
        )
        sd_sequence_length = (
            data_batch["{}_{}_length".format(x_str, x_fields[0])] - 2
        )
        sd_enc_outputs, _ = x_encoder(
            sd_embeds, sequence_length=sd_sequence_length
        )
        sd_enc_outputs = concat_encoder_outputs(sd_enc_outputs)

        return (
            sent_ids,
            sent_embeds,
            sent_enc_outputs,
            sent_sequence_length,
            sd_ids,
            sd_embeds,
            sd_enc_outputs,
            sd_sequence_length,
        )

    encode_results = [encode(ref_str) for ref_str in range(2)]
    (
        sent_ids,
        sent_embeds,
        sent_enc_outputs,
        sent_sequence_length,
        sd_ids,
        sd_embeds,
        sd_enc_outputs,
        sd_sequence_length,
    ) = zip(*encode_results)

    # get rnn cell
    rnn_cell = tx.core.layers.get_rnn_cell(Config.config_model.rnn_cell)

    def get_decoder(
        cell, y__ref_flag, x_ref_flag, tgt_ref_flag, beam_width=None
    ):
        output_layer_params = (
            {"output_layer": tf.identity}
            if Config.copy_flag
            else {"vocab_size": vocab.size}
        )

        if Config.attn_flag:  # attention
            if Config.attn_x and Config.attn_y_:
                memory = tf.concat(
                    [sent_enc_outputs[y__ref_flag], sd_enc_outputs[x_ref_flag]],
                    axis=1,
                )
                memory_sequence_length = None
            elif Config.attn_y_:
                memory = sent_enc_outputs[y__ref_flag]
                memory_sequence_length = sent_sequence_length[y__ref_flag]
            elif Config.attn_x:
                memory = sd_enc_outputs[x_ref_flag]
                memory_sequence_length = sd_sequence_length[x_ref_flag]
            else:
                raise Exception(
                    "Must specify either y__ref_flag or x_ref_flag."
                )
            attention_decoder = tx.modules.AttentionRNNDecoder(
                cell=cell,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
                hparams=Config.config_model.attention_decoder,
                **output_layer_params
            )
            if not Config.copy_flag:
                return attention_decoder
            cell = (
                attention_decoder.cell
                if beam_width is None
                else attention_decoder._get_beam_search_cell(beam_width)
            )

        if Config.copy_flag:  # copynet
            kwargs = {
                "y__ids": sent_ids[y__ref_flag][:, 1:],
                "y__states": sent_enc_outputs[y__ref_flag][:, 1:],
                "y__lengths": sent_sequence_length[y__ref_flag] - 1,
                "x_ids": sd_ids[x_ref_flag]["value"],
                "x_states": sd_enc_outputs[x_ref_flag],
                "x_lengths": sd_sequence_length[x_ref_flag],
            }

            if tgt_ref_flag is not None:
                kwargs.update(
                    {
                        "input_ids": data_batch[
                            "{}_text_ids".format(y_strs[tgt_ref_flag])
                        ][:, :-1]
                    }
                )

            memory_prefixes = []

            if Config.copy_y_:
                memory_prefixes.append("y_")

            if Config.copy_x:
                memory_prefixes.append("x")

            if beam_width is not None:
                kwargs = {
                    name: tile_batch(value, beam_width)
                    for name, value in kwargs.items()
                }

            def get_get_copy_scores(memory_ids_states_lengths, output_size):
                memory_copy_states = [
                    tf.layers.dense(
                        memory_states,
                        units=output_size,
                        activation=None,
                        use_bias=False,
                    )
                    for _, memory_states, _ in memory_ids_states_lengths
                ]

                def get_copy_scores(query, coverities=None):
                    ret = []

                    if Config.copy_y_:
                        memory = memory_copy_states[len(ret)]
                        if coverities is not None:
                            memory = memory + tf.layers.dense(
                                coverities[len(ret)],
                                units=output_size,
                                activation=None,
                                use_bias=False,
                            )
                        memory = tf.nn.tanh(memory)
                        ret_y_ = tf.einsum("bim,bm->bi", memory, query)
                        ret.append(ret_y_)

                    if Config.copy_x:
                        memory = memory_copy_states[len(ret)]
                        if coverities is not None:
                            memory = memory + tf.layers.dense(
                                coverities[len(ret)],
                                units=output_size,
                                activation=None,
                                use_bias=False,
                            )
                        memory = tf.nn.tanh(memory)
                        ret_x = tf.einsum("bim,bm->bi", memory, query)
                        ret.append(ret_x)

                    return ret

                return get_copy_scores

            covrity_dim = (
                Config.config_model.coverage_state_dim
                if Config.coverage
                else None
            )
            coverity_rnn_cell_hparams = (
                Config.config_model.coverage_rnn_cell
                if Config.coverage
                else None
            )
            cell = CopyNetWrapper(
                cell=cell,
                vocab_size=vocab.size,
                memory_ids_states_lengths=[
                    tuple(
                        kwargs["{}_{}".format(prefix, s)]
                        for s in ("ids", "states", "lengths")
                    )
                    for prefix in memory_prefixes
                ],
                input_ids=kwargs["input_ids"]
                if tgt_ref_flag is not None
                else None,
                get_get_copy_scores=get_get_copy_scores,
                coverity_dim=covrity_dim,
                coverity_rnn_cell_hparams=coverity_rnn_cell_hparams,
                disabled_vocab_size=Config.disabled_vocab_size,
                eps=Config.eps,
            )

        decoder = tx.modules.BasicRNNDecoder(
            cell=cell,
            hparams=Config.config_model.decoder,
            **output_layer_params
        )
        return decoder

    def get_decoder_and_outputs(
        cell, y__ref_flag, x_ref_flag, tgt_ref_flag, params, beam_width=None
    ):
        decoder = get_decoder(
            cell, y__ref_flag, x_ref_flag, tgt_ref_flag, beam_width=beam_width
        )
        if beam_width is None:
            ret = decoder(**params)
        else:
            ret = tx.modules.beam_search_decode(
                decoder_or_cell=decoder, beam_width=beam_width, **params
            )
        return (decoder,) + ret

    get_decoder_and_outputs = tf.make_template(
        "get_decoder_and_outputs", get_decoder_and_outputs
    )

    def teacher_forcing(cell, y__ref_flag, x_ref_flag, loss_name):
        tgt_ref_flag = x_ref_flag
        tgt_str = y_strs[tgt_ref_flag]
        sequence_length = data_batch["{}_length".format(tgt_str)] - 1
        decoder, tf_outputs, final_state, _ = get_decoder_and_outputs(
            cell,
            y__ref_flag,
            x_ref_flag,
            tgt_ref_flag,
            {
                "decoding_strategy": "train_greedy",
                "inputs": sent_embeds[tgt_ref_flag],
                "sequence_length": sequence_length,
            },
        )

        tgt_sent_ids = data_batch["{}_text_ids".format(tgt_str)][:, 1:]
        loss = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=tgt_sent_ids,
            logits=tf_outputs.logits,
            sequence_length=sequence_length,
            average_across_batch=False,
        )
        if (
            Config.add_bleu_weight
            and y__ref_flag is not None
            and tgt_ref_flag is not None
            and y__ref_flag != tgt_ref_flag
        ):
            w = tf.py_func(
                batch_bleu,
                [sent_ids[y__ref_flag], tgt_sent_ids],
                tf.float32,
                stateful=False,
                name="W_BLEU",
            )
            w.set_shape(loss.get_shape())
            loss = w * loss
        loss = tf.reduce_mean(loss, 0)

        if Config.copy_flag and Config.exact_cover_w != 0:
            sum_copy_probs = list(
                map(
                    lambda t: tf.cast(t, tf.float32), final_state.sum_copy_probs
                )
            )
            memory_lengths = [
                lengths
                for _, _, lengths in decoder.cell.memory_ids_states_lengths
            ]
            exact_coverage_losses = [
                tf.reduce_mean(
                    tf.reduce_sum(
                        tx.utils.mask_sequences(
                            tf.square(sum_copy_prob - 1.0), memory_length
                        ),
                        1,
                    )
                )
                for sum_copy_prob, memory_length in zip(
                    sum_copy_probs, memory_lengths
                )
            ]
            print_xe_loss_op = tf.print(loss_name, "xe loss:", loss)
            with tf.control_dependencies([print_xe_loss_op]):
                for i, exact_coverage_loss in enumerate(exact_coverage_losses):
                    print_op = tf.print(
                        loss_name,
                        "exact coverage loss {:d}:".format(i),
                        exact_coverage_loss,
                    )
                    with tf.control_dependencies([print_op]):
                        loss += Config.exact_cover_w * exact_coverage_loss

        losses[loss_name] = loss

        return decoder, tf_outputs, loss

    def beam_searching(cell, y__ref_flag, x_ref_flag, beam_width):
        start_tokens = (
            tf.ones_like(data_batch["y_aux_length"]) * vocab.bos_token_id
        )
        end_token = vocab.eos_token_id

        decoder, bs_outputs, _, _ = get_decoder_and_outputs(
            cell,
            y__ref_flag,
            x_ref_flag,
            None,
            {
                "embedding": embedders["y_aux"],
                "start_tokens": start_tokens,
                "end_token": end_token,
                "max_decoding_length": Config.config_train.infer_max_decoding_length,
            },
            beam_width=Config.config_train.infer_beam_width,
        )

        return decoder, bs_outputs

    decoder, tf_outputs, loss = teacher_forcing(rnn_cell, 1, 0, "MLE")
    rec_decoder, _, rec_loss = teacher_forcing(rnn_cell, 1, 1, "REC")
    rec_weight = Config.rec_w

    step_stage = tf.cast(step, tf.float32) / tf.constant(800.0)
    rec_weight = tf.case(
        [
            (
                tf.less_equal(step_stage, tf.constant(1.0)),
                lambda: tf.constant(1.0),
            ),
            (tf.greater(step_stage, tf.constant(2.0)), lambda: Config.rec_w),
        ],
        default=lambda: tf.constant(1.0)
        - (step_stage - 1) * (1 - Config.rec_w),
    )
    joint_loss = (1 - rec_weight) * loss + rec_weight * rec_loss
    losses["joint"] = joint_loss

    tiled_decoder, bs_outputs = beam_searching(
        rnn_cell, 1, 0, Config.config_train.infer_beam_width
    )

    train_ops = {
        name: get_train_op(
            losses[name], hparams=Config.config_train.train[name]
        )
        for name in Config.config_train.train
    }

    return train_ops, bs_outputs


class Rewriter:
    def __init__(self):
        self.sess = tf.Session()
        # data batch
        self.datasets = {
            mode: tx.data.MultiAlignedData(hparams)
            for mode, hparams in Config.config_data.datas.items()
        }
        self.data_iterator = tx.data.FeedableDataIterator(self.datasets)
        self.data_batch = self.data_iterator.get_next()

        self.global_step = tf.train.get_or_create_global_step()

        self.train_ops, self.bs_outputs = build_model(
            self.data_batch, self.datasets["train"], self.global_step
        )

        self.summary_ops = {
            name: tf.summary.merge(
                tf.get_collection(
                    tf.GraphKeys.SUMMARIES,
                    scope=get_scope_name_of_train_op(name),
                ),
                name=get_scope_name_of_summary_op(name),
            )
            for name in self.train_ops.keys()
        }

        self.saver = tf.train.Saver(max_to_keep=None)

        # global best_ever_val_bleu
        self.best_ever_val_bleu = 0.0

    def save_to(self, directory, step):
        print("saving to {} ...".format(directory))

        saved_path = self.saver.save(self.sess, directory, global_step=step)

        print("saved to {}".format(saved_path))

    def restore_from_path(self, ckpt_path):
        print("restoring from {} ...".format(ckpt_path))

        try:
            self.saver.restore(self.sess, ckpt_path)
        except tf.errors.NotFoundError:
            print("Some variables are missing. Try optimistically restoring.")
            (get_optimistic_saver(ckpt_path)).restore(self.sess, ckpt_path)

        print("done.")

    def restore_from(self, directory):
        if os.path.exists(directory):
            ckpt_path = tf.train.latest_checkpoint(directory)
            self.restore_from_path(ckpt_path)

        else:
            print("cannot find checkpoint directory {}".format(directory))

    def train_epoch(self, sess, summary_writer, mode, train_op, summary_op):
        print("in _train_epoch")

        self.data_iterator.restart_dataset(sess, mode)

        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            self.data_iterator.handle: self.data_iterator.get_handle(
                sess, mode
            ),
        }

        while True:
            try:
                loss, summary = sess.run((train_op, summary_op), feed_dict)

                step = tf.train.global_step(sess, self.global_step)

                print("step {:d}: loss = {:.6f}".format(step, loss))

                summary_writer.add_summary(summary, step)

                # if step % config_train.steps_per_eval == 0:
                #     _eval_epoch(sess, summary_writer, 'val')

            except tf.errors.OutOfRangeError:
                break

        print("end _train_epoch")

    def eval_epoch(self, mode):

        # As discussed in this thread, to use this model in an multi-thread
        # environment (such as a web server). We need to explicitly use the
        # default session:
        # https://github.com/keras-team/keras/issues/2397#issuecomment-254919212
        with self.sess.graph.as_default():
            print("in _eval_epoch with mode {}".format(mode))

            self.data_iterator.restart_dataset(self.sess, mode)

            feed_dict = {
                self.data_iterator.handle: self.data_iterator.get_handle(
                    self.sess, mode
                ),
                tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            }

            step = tf.train.global_step(self.sess, self.global_step)

            ref_hypo_pairs = []
            fetches = [
                [self.data_batch["y_aux_text"], self.data_batch["y_ref_text"]],
                [
                    self.data_batch["x_value_text"],
                    self.data_batch["x_ref_value_text"],
                ],
                self.bs_outputs.predicted_ids,
            ]

            if not os.path.exists(Config.dir_model):
                os.makedirs(Config.dir_model)

            hypo_file_name = os.path.join(
                Config.dir_model, "hypos.step{}.{}.txt".format(step, mode)
            )
            hypo_file = open(hypo_file_name, "w")

            cnt = 0
            while True:
                try:
                    target_texts, entry_texts, output_ids = self.sess.run(
                        fetches, feed_dict
                    )
                    target_texts = [
                        tx.utils.strip_special_tokens(
                            texts[:, 1:].tolist(), is_token_list=True
                        )
                        for texts in target_texts
                    ]
                    entry_texts = [
                        tx.utils.strip_special_tokens(
                            texts[:, 1:].tolist(), is_token_list=True
                        )
                        for texts in entry_texts
                    ]

                    output_ids = output_ids[:, :, 0]
                    output_texts = tx.utils.map_ids_to_strs(
                        ids=output_ids.tolist(),
                        vocab=self.datasets[mode].vocab("y_aux"),
                        join=False,
                    )

                    target_texts = list(zip(*target_texts))
                    entry_texts = list(zip(*entry_texts))
                    for ref, hypo in zip(target_texts, output_texts):
                        if cnt < 10:
                            print("cnt = {}".format(cnt))
                            for i, s in enumerate(ref):
                                print("ref{}: {}".format(i, " ".join(s)))
                            print("hypo: {}".format(" ".join(hypo)))
                            return "{}".format(" ".join(hypo))
                        print(" ".join(hypo), file=hypo_file)
                        cnt += 1
                    print("processed {} samples".format(cnt))

                    ref_hypo_pairs.extend(
                        zip(target_texts, entry_texts, output_texts)
                    )

                except tf.errors.OutOfRangeError:
                    break

            hypo_file.close()

            refs, entrys, hypos = zip(*ref_hypo_pairs)

            bleus = []
            get_bleu_name = "{}_BLEU".format
            for i in range(1, 2):
                refs_ = list(map(lambda ref: ref[i : i + 1], refs))
                ents_ = list(map(lambda ent: ent[i : i + 1], entrys))
                entrys = list(zip(*entrys))
                bleu = corpus_bleu(refs_, hypos)
                bleus.append(bleu)

            summary = tf.Summary()
            for i, bleu in enumerate(bleus):
                summary.value.add(
                    tag="{}/{}".format(mode, get_bleu_name(i)),
                    simple_value=bleu,
                )

            self.summary_writer.add_summary(summary, step)
            self.summary_writer.flush()

            bleu = bleus[0]
            if mode == "val":
                if bleu > self.best_ever_val_bleu:
                    self.best_ever_val_bleu = bleu
                    print("updated best val bleu: {}".format(bleu))

                    self.save_to(Config.ckpt_best, step)

            print("end _eval_epoch")
            return

    def load_model(self):

        # As discussed in this thread, to use this model in an multi-thread
        # environment (such as a web server). We need to explicitly use the
        # default session:
        # https://github.com/keras-team/keras/issues/2397#issuecomment-254919212
        with self.sess.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(tf.tables_initializer())
            # self.sess.run(self.data_iterator)

            print("loading model ", Config.restore_from, Config.dir_model)

            if Config.restore_from:
                self.restore_from_path(Config.restore_from)
            else:
                self.restore_from(Config.dir_model)

            self.summary_writer = tf.summary.FileWriter(
                Config.dir_summary, self.sess.graph, flush_secs=30
            )

            epoch = 0
            while epoch < Config.config_train.max_epochs:
                name = "joint"
                train_op = self.train_ops[name]
                summary_op = self.summary_ops[name]

                step = tf.train.global_step(self.sess, self.global_step)

                self.train_epoch(
                    self.sess,
                    self.summary_writer,
                    "train",
                    train_op,
                    summary_op,
                )

                epoch += 1

                step = tf.train.global_step(self.sess, self.global_step)
                self.save_to(Config.ckpt_model, step)
