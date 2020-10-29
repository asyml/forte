# Copyright 2020 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import tensorflow as tf
import texar.tf as tx

# pylint: disable=no-name-in-module
from forte.models.imdb_text_classifier.utils import data_utils, model_utils


class IMDBClassifier:
    """
    A baseline text classifier for the IMDB dataset.
    The input data should be CSV format with columns (content label id).
    """

    def __init__(self, config_data, config_classifier, checkpoint=None, pretrained_model_name="bert-base-uncased"):
        """Constructs the text classifier.
        Args:
            config_data: string, data config file.
        """
        self.config_data = config_data
        self.config_classifier = config_classifier
        self.checkpoint = checkpoint
        self.pretrained_model_name = pretrained_model_name
    
    def prepare_data(self, csv_data_dir):
        """Prepares data.
        """
        tf.logging.info("Loading data")

        if self.config_data.tfrecord_data_dir is None:
            tfrecord_output_dir = csv_data_dir
        else:
            tfrecord_output_dir = self.config_data.tfrecord_data_dir
        tx.utils.maybe_create_dir(tfrecord_output_dir)
        
        processor = data_utils.IMDbProcessor()

        num_classes = len(processor.get_labels())
        num_train_data = len(processor.get_train_examples(csv_data_dir))
        tf.logging.info(
            'num_classes:%d; num_train_data:%d' % (num_classes, num_train_data))

        tokenizer = tx.data.BERTTokenizer(
            pretrained_model_name=self.pretrained_model_name)

        # Produces TFRecord files
        data_utils.prepare_TFRecord_data(
            processor=processor,
            tokenizer=tokenizer,
            data_dir=csv_data_dir,
            max_seq_length=self.config_data.max_seq_length,
            output_dir=tfrecord_output_dir)

    def run(self, do_train, do_eval, do_test, output_dir="output/"):
        """
        Builds the model and runs.
        """
        tf.logging.set_verbosity(tf.logging.INFO)

        tx.utils.maybe_create_dir(output_dir)

        # Loads data
        num_train_data = self.config_data.num_train_data

        train_dataset = tx.data.TFRecordData(hparams=self.config_data.train_hparam)
        eval_dataset = tx.data.TFRecordData(hparams=self.config_data.eval_hparam)
        test_dataset = tx.data.TFRecordData(hparams=self.config_data.test_hparam)

        iterator = tx.data.FeedableDataIterator({
            'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset})
        batch = iterator.get_next()
        input_ids = batch["input_ids"]
        segment_ids = batch["segment_ids"]
        batch_size = tf.shape(input_ids)[0]
        input_length = tf.reduce_sum(1 - tf.cast(tf.equal(input_ids, 0), tf.int32),
                                    axis=1)
        # Builds BERT
        hparams = {
            'clas_strategy': 'cls_time'
        }
        model = tx.modules.BERTClassifier(
            pretrained_model_name=self.pretrained_model_name,
            hparams=hparams)
        logits, preds = model(input_ids, input_length, segment_ids)

        accu = tx.evals.accuracy(batch['label_ids'], preds)

        # Optimization
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=batch["label_ids"], logits=logits)
        global_step = tf.Variable(0, trainable=False)

        # Builds learning rate decay scheduler
        static_lr = self.config_classifier.lr['static_lr']
        num_train_steps = int(num_train_data / self.config_data.train_batch_size
                            * self.config_data.max_train_epoch)
        num_warmup_steps = int(num_train_steps * self.config_data.warmup_proportion)
        lr = model_utils.get_lr(global_step, num_train_steps,  # lr is a Tensor
                                num_warmup_steps, static_lr)

        opt = tx.core.get_optimizer(
            global_step=global_step,
            learning_rate=lr,
            hparams=self.config_classifier.opt
        )

        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=global_step,
            learning_rate=None,
            optimizer=opt)

        # Train/eval/test routine

        def _train_epoch(sess):
            """Trains on the training set, and evaluates on the dev set
            periodically.
            """
            iterator.restart_dataset(sess, 'train')

            fetches = {
                'train_op': train_op,
                'loss': loss,
                'batch_size': batch_size,
                'step': global_step
            }

            while True:
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, 'train'),
                        tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
                    }
                    rets = sess.run(fetches, feed_dict)
                    step = rets['step']

                    dis_steps = self.config_data.display_steps
                    if dis_steps > 0 and step % dis_steps == 0:
                        tf.logging.info('step:%d; loss:%f;' % (step, rets['loss']))

                    eval_steps = self.config_data.eval_steps
                    if eval_steps > 0 and step % eval_steps == 0:
                        _eval_epoch(sess)

                except tf.errors.OutOfRangeError:
                    break

        def _eval_epoch(sess):
            """Evaluates on the dev set.
            """
            iterator.restart_dataset(sess, 'eval')

            cum_acc = 0.0
            cum_loss = 0.0
            nsamples = 0
            fetches = {
                'accu': accu,
                'loss': loss,
                'batch_size': batch_size,
            }
            while True:
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, 'eval'),
                        tx.context.global_mode(): tf.estimator.ModeKeys.EVAL,
                    }
                    rets = sess.run(fetches, feed_dict)

                    cum_acc += rets['accu'] * rets['batch_size']
                    cum_loss += rets['loss'] * rets['batch_size']
                    nsamples += rets['batch_size']
                except tf.errors.OutOfRangeError:
                    break

            tf.logging.info('eval accu: {}; loss: {}; nsamples: {}'.format(
                cum_acc / nsamples, cum_loss / nsamples, nsamples))

        def _test_epoch(sess):
            """Does predictions on the test set.
            """
            iterator.restart_dataset(sess, 'test')

            _all_preds = []
            while True:
                try:
                    feed_dict = {
                        iterator.handle: iterator.get_handle(sess, 'test'),
                        tx.context.global_mode(): tf.estimator.ModeKeys.PREDICT,
                    }
                    _preds = sess.run(preds, feed_dict=feed_dict)
                    _all_preds.extend(_preds.tolist())
                except tf.errors.OutOfRangeError:
                    break

            output_file = os.path.join(output_dir, "test_results.tsv")
            with tf.gfile.GFile(output_file, "w") as writer:
                writer.write('\n'.join(str(p) for p in _all_preds))

        session_config = tf.ConfigProto()

        with tf.Session(config=session_config) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())

            # Restores trained model if specified
            saver = tf.train.Saver()
            if self.checkpoint:
                saver.restore(sess, self.checkpoint)

            iterator.initialize_dataset(sess)

            if do_train:
                for i in range(self.config_data.max_train_epoch):
                    _train_epoch(sess)
                saver.save(sess, output_dir + '/model.ckpt')

            if do_eval:
                _eval_epoch(sess)

            if do_test:
                _test_epoch(sess)
        
