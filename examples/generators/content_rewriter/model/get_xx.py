from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import pickle

import tensorflow as tf

from examples.generators.content_rewriter.model.utils_e2e_clean import *

# pylint: disable=invalid-name, no-member, too-many-locals


inf = int(1e9)
def calc_cost(a, b):
    if a.attribute != b.attribute:
        return inf
    if a.entry.isdigit():
        if b.entry.isdigit():
            return abs(int(a.entry) - int(b.entry))
        else:
            return inf
    else:
        if b.entry.isdigit():
            return inf
        else:
            return 0 if a.entry == b.entry else 1


def get_match(text00, text01, text02, text10, text11, text12):
    text00, text01, text02, text10, text11, text12 = map(
        strip_special_tokens_of_list,
        (text00, text01, text02, text10, text11, text12))
    texts = [DataItem(text00, text01, text02),
             DataItem(text10, text11, text12)]
    xs = list(map(pack_sd, texts))

    cost = [[calc_cost(x_i, x_j) for x_j in xs[1]] for x_i in xs[0]]
    matches = []
    for idx, cost_i in enumerate(cost):
        if min(cost_i) == inf:
            match = (idx, -1)
        else:
            match = (idx, cost_i.index(min(cost_i)))
        matches.append(match)
    return matches

    # return Munkres().compute(cost)


batch_get_match = batchize(get_match)


def main():
    # data batch
    datasets = {mode: tx.data.MultiAlignedData(hparams)
                for mode, hparams in config_data.datas.items()}
    data_iterator = tx.data.FeedableDataIterator(datasets)
    data_batch = data_iterator.get_next()


    def _get_match(sess, mode):
        print('in _get_match')

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            data_iterator.handle: data_iterator.get_handle(sess, mode),
        }

        with open('match.{}.pkl'.format(mode), 'wb') as out_file:
            while True:
                try:
                    batch = sess.run(data_batch, feed_dict)
                    texts = [[batch['{}{}_text'.format(field, ref_str)]
                              for field in sd_fields]
                             for ref_str in ref_strs]
                    matches = batch_get_match(*(texts[0] + texts[1]))
                    for match in matches:
                        pickle.dump(match, out_file)

                except tf.errors.OutOfRangeError:
                    break

        print('end _get_match')


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        _get_match(sess, 'test')
        _get_match(sess, 'valid')
        _get_match(sess, 'train')


if __name__ == '__main__':
    flags = tf.flags
    flags.DEFINE_string("config_data", "config_data_nba_stable",
                        "The data config.")
    flags.DEFINE_boolean("verbose", False, "verbose.")
    FLAGS = flags.FLAGS

    config_data = importlib.import_module(FLAGS.config_data)

    main()
