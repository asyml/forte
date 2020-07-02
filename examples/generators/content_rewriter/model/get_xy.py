from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import pickle

import numpy as np
import tensorflow as tf

from examples.generators.content_rewriter.model.utils_e2e_clean import *

# pylint: disable=invalid-name, no-member, too-many-locals

"""""
with open(os.path.join('data2text', 'res', 'train.idx.json'), 'r') as idx_f:
    sent_idx = json.load(idx_f)
sent_to_idx = dict(map(lambda pair: (' ', pair[1]), sent_idx))
"""


def get_align(text00, text01, text02, text1):
    #    print('=========text1 is  :{}'.format(text1))
    text00, text01, text02, text1 = map(
        strip_special_tokens_of_list,
        (text00, text01, text02, text1))
    sd_texts, sent_texts = pack_sd(DataItem(text00, text01, text02)), text1
    # print('=========sent_texts is  :{}'.format(sent_texts))
    sent = ' '.join(sent_texts)
    # print('=========sent is  :{}'.format(sent))
    idxs = []
    for entry in text00:
        idxs.append(text1.tolist().index(entry) if entry in text1 else -1)

    assert len(idxs) == len(
        sd_texts), "\nidxs = {}\nsd_texts = {}\nsent = {}".format(idxs,
                                                                  sd_texts,
                                                                  sent)

    align = [
        [int(j == idx)
         for j in range(len(sent_texts))]
        for idx in idxs]
    return np.array(align)


batch_get_align = batchize(get_align)


def print_align(sd_text0, sd_text1, sd_text2, sent_text, align):
    sd_text = [sd_text0, sd_text1, sd_text2]
    for text, name in zip(sd_text, sd_fields):
        print('{:>20}'.format(name) + ' '.join(map('{:>18}'.format, text)))
    for j, sent_token in enumerate(sent_text):
        print('{:>20}'.format(sent_token) + ' '.join(map(
            lambda x: '{:18}'.format(x) if x != 0 else ' ' * 18,
            align[:, j])))


batch_print_align = batchize(print_align)


def main():
    # data batch
    datasets = {mode: tx.data.MultiAlignedData(hparams)
                for mode, hparams in config_data.datas.items()}
    data_iterator = tx.data.FeedableDataIterator(datasets)
    data_batch = data_iterator.get_next()

    def _get_align(sess, mode):
        print('in _get_align')

        data_iterator.restart_dataset(sess, mode)
        feed_dict = {
            tx.global_mode(): tf.estimator.ModeKeys.EVAL,
            data_iterator.handle: data_iterator.get_handle(sess, mode),
        }

        with open('align.pkl', 'wb') as out_file:
            while True:
                try:
                    batch = sess.run(data_batch, feed_dict)
                    sd_texts, sent_texts = (
                        [batch['{}{}_text'.format(field, ref_strs[1])]
                         for field in fields]
                        for fields in (sd_fields, sent_fields))
                    aligns = batch_get_align(*(sd_texts + sent_texts))
                    sd_texts, sent_texts = (
                        [batch_strip_special_tokens_of_list(texts)
                         for texts, field in zip(all_texts, fields)]
                        for all_texts, fields in zip(
                        (sd_texts, sent_texts), (sd_fields, sent_fields)))
                    if FLAGS.verbose:
                        batch_print_align(*(sd_texts + sent_texts + [aligns]))
                    for align in aligns:
                        pickle.dump(align, out_file)

                except tf.errors.OutOfRangeError:
                    break

        print('end _get_align')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        _get_align(sess, 'train')


if __name__ == '__main__':
    flags = tf.flags
    flags.DEFINE_string("config_data", "config_data_nba_stable",
                        "The data config.")
    flags.DEFINE_boolean("verbose", False, "verbose.")
    FLAGS = flags.FLAGS

    config_data = importlib.import_module(FLAGS.config_data)

    main()
