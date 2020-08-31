######################################################################
#  CliNER - model.py                                                 #
#                                                                    #
#  Willie Boag                                                       #
#                                                                    #
#  Purpose: Define the model for clinical concept extraction.        #
######################################################################

import copy
import io
import os
import pickle
import random
import shutil
import sys
from time import localtime, strftime

import numpy as np
from sklearn.feature_extraction import DictVectorizer

import tensorflow as tf

from examples.Cliner.CliNER.code import DatasetCliner_experimental as Exp
from examples.Cliner.CliNER.code import entity_lstm as entity_model
from examples.Cliner.CliNER.code import helper_dataset as hd
from examples.Cliner.CliNER.code import training_predict_LSTM
from examples.Cliner.CliNER.code.feature_extraction.features import \
    extract_features
from examples.Cliner.CliNER.code.machine_learning import crf
from examples.Cliner.CliNER.code.notes.documents import labels as tag2id, id2tag
from examples.Cliner.CliNER.code.tools import flatten, save_list_structure, \
    reconstruct_list
from examples.Cliner.CliNER.code.tools import print_str, print_vec, \
    print_files, \
    write

cliner_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tmp_dir = os.path.join(cliner_dir, 'data', 'tmp')


class ClinerModel:
    def log(self, out, model_file=None):
        '''
        ClinerModel::log()
        Log training information of model.
        @param out.         Either a filename or file channel to output the
        log string.
        @param model_file.  A path to optionally identify where the model was
        saved.
        @return None
        '''
        if not self._log:
            log = self.__log_str(model_file)
        else:
            log = self._log

        # depending on whether it is already opened as a channel
        if isinstance(out, type(sys.stdout)):
            write(out, '%s\n' % log)
        else:
            with open(out, 'a') as f:
                write(f, '%s\n' % log)

    def __log_str_NEURAL(self, model_file=None):
        ""

    def __log_str(self, model_file=None):
        '''
        ClinerModel::__log_str()
        Build a string of information about training for the model's log file.
        @param model_file.  A path to optionally identify where the model was
        saved.
        @return  A string of the model's training information
        '''

        assert self._is_trained, 'ClinerModel not trained'
        with io.StringIO() as f:
            write(f, u'\n')
            write(f, '-' * 40)
            write(f, u'\n\n')
            if model_file:
                write(f, 'model    : %s\n' % os.path.abspath(model_file))
                write(f, u'\n')

            if self._use_lstm:
                write(f, u'modeltype: LSTM\n')
            else:
                write(f, u'modeltype: CRF\n')

            if 'hyperparams' in self._score:
                for name, value in self._score['hyperparams'].items():
                    write(f, u'\t%-10s: %s\n' % (name, value))
            write(f, u'\n')

            print_str(f, 'features', self._features)
            write(f, u'\n')

            write(f, u'\n')
            write(f, 'training began: %s\n' % self._time_train_begin)
            write(f, 'training ended: %s\n' % self._time_train_end)
            write(f, u'\n')

            write(f, u'scores\n')
            print_vec(f, 'train precision', self._score['train']['precision'])
            print_vec(f, 'train recall   ', self._score['train']['recall'])
            print_vec(f, 'train f1       ', self._score['train']['f1'])
            write(f, self._score['train']['conf'])

            if 'dev' in self._score:
                print_vec(f, u'dev precision   ',
                          self._score['dev']['precision'])
                print_vec(f, u'dev recall      ', self._score['dev']['recall'])
                print_vec(f, u'dev f1          ', self._score['dev']['f1'])
                write(f, self._score['dev']['conf'])

            if 'test' in self._score:
                print_vec(f, u'test precision   ',
                          self._score['test']['precision'])
                print_vec(f, u'test recall      ',
                          self._score['test']['recall'])
                print_vec(f, u'test f1          ', self._score['test']['f1'])
                write(f, self._score['test']['conf'])

            if 'history' in self._score:
                for label, vec in self._score['history'].items():
                    print_vec(f, '%-16s' % label, vec)
                write(f, u'\n')

            if self._training_files:
                write(f, u'\n')
                write(f, u'Training Files\n')
                if len(self._training_files) < 200:
                    print_files(f, self._training_files)
                else:
                    write(f, '\t%d files\n' % len(self._training_files))
                write(f, u'\n')

            write(f, u'-' * 40)
            write(f, u'\n\n')

            # get output as full string
            contents = f.getvalue()
        return contents

    def __init__(self, use_lstm):
        """
        ClinerModel::__init__()

        Instantiate a ClinerModel object.

        @param use_lstm. Bool indicating whether to train a CRF or LSTM.
        """

        self._use_lstm = use_lstm
        self._is_trained = False
        self._clf = "latin1"
        self._vocab = None
        self._training_files = None
        self._log = None
        self._text_feats = None

        # Import the tools for either CRF or LSTM
        if use_lstm:
            # NEW

            self._pretrained_dataset = None
            self._pretrained_wordvectors = None

            self._current_model = None
            self._parameters = None

    # pylint: disable=dangerous-default-value
    def train(self, train_notes, val=[], test=[]):
        """
        ClinerModel::train()

        Purpose: Train a Machine Learning model on annotated data

        @param notes. A list of Note objects (containing text and annotations)
        @return       None
        """

        # Extract formatted data
        train_sents = flatten([n.getTokenizedSentences() for n in train_notes])
        train_labels = flatten([n.getTokenLabels() for n in train_notes])

        if test:
            test_sents = flatten([n.getTokenizedSentences() for n in test])
            test_labels = flatten([n.getTokenLabels() for n in test])
        else:
            test_sents = []
            test_labels = []

        if val:
            print("VAL")
            val_sents = flatten([n.getTokenizedSentences() for n in val])
            val_labels = flatten([n.getTokenLabels() for n in val])
            self.train_fit(train_sents,
                           train_labels,
                           val_sents=val_sents,
                           val_labels=val_labels,
                           test_sents=test_sents,
                           test_labels=test_labels)

        else:
            print("NO DEV")
            self.train_fit(train_sents,
                           train_labels,
                           dev_split=0.1,
                           test_sents=test_sents,
                           test_labels=test_labels)

        # pylint: disable=attribute-defined-outside-init
        self._train_files = [n.getName() for n in train_notes + val]

    def train_fit(self,
                  train_sents,
                  train_labels,
                  val_sents=None,
                  val_labels=None,
                  test_sents=None,
                  test_labels=None,
                  dev_split=None):
        """
        ClinerModel::train_fit()

        Purpose: Train clinical concept extraction model using annotated data.

        @param train_sents. A list of sentences, where each sentence is
        tokenized into words.
        @param train_labels. Parallel to 'train_sents', 7-way labels for
        concept spans.
        @param val_sents.   Validation data. Same format as tokenized_sents
        @param val_labels.  Validation data. Same format as iob_nested_labels
        @param dev_split    A real number from 0 to 1
        """

        # metadata
        # pylint: disable=attribute-defined-outside-init
        self._time_train_begin = strftime("%Y-%m-%d %H:%M:%S", localtime())

        # train classifier
        if not self._use_lstm:
            # pylint: disable=unbalanced-tuple-unpacking
            voc, clf, dev_score, enabled_features = generic_train(
                'all',
                train_sents,
                train_labels,
                self._use_lstm,
                val_sents=val_sents,
                val_labels=val_labels,
                test_sents=test_sents,
                test_labels=test_labels,
                dev_split=dev_split)
            # pylint: disable=attribute-defined-outside-init
            self._is_trained = True
            self._vocab = voc
            self._clf = clf
            self._score = dev_score
            self._features = enabled_features
            # metadata
            self._time_train_end = strftime("%Y-%m-%d %H:%M:%S", localtime())

        else:
            print("IN ERROR CHECK")
            print(dev_split)
            parameters, dataset, best = generic_train('all',
                                                      train_sents,
                                                      train_labels,
                                                      self._use_lstm,
                                                      val_sents=val_sents,
                                                      val_labels=val_labels,
                                                      test_sents=test_sents,
                                                      test_labels=test_labels,
                                                      dev_split=dev_split)
            # pylint: disable=attribute-defined-outside-init
            self._is_trained = True
            self.pretrained_dataset = dataset
            self.parameters = parameters
            self._score = best
            self._time_train_end = strftime("%Y-%m-%d %H:%M:%S", localtime())
            print("BEST EPOCH")
            print(best)
        # self._vocab = voc
        # self._clf   = clf
        # self._score = dev_score
        # self._features = enabled_features

        # metadata
        # self._time_train_end = strftime("%Y-%m-%d %H:%M:%S", localtime())

    def predict_classes_from_document(self, document):
        """
        ClinerModel::predict_classes_from_documents()

        Predict concept annotations for a given document

        @param note. A Document object (containing text and annotations)
        @return      List of predictions
        """
        # Extract formatted data
        tokenized_sents = document.getTokenizedSentences()

        return self.predict_classes(tokenized_sents)

    def predict_classes(self, tokenized_sents):
        """
        ClinerModel::predict_classes()

        Predict concept annotations for unlabeled, tokenized sentences

        @param tokenized_sents. A list of sentences, where each sentence is
        tokenized
                                  into words
        @return                  List of predictions
        """

        # Predict labels for prose
        vectorized_pred = generic_predict('all',
                                          tokenized_sents,
                                          vocab=self._vocab,
                                          clf=self._clf,
                                          use_lstm=self._use_lstm)
        # pretrained_dataset=self._pretrained_dataset,
        # tokens_to_vec=self._pretrained_wordvector,
        # current_model=self._current_model,
        # parameters=self.parameters)

        # self._current_model=model

        if self._use_lstm:
            iob_pred = vectorized_pred
        else:
            iob_pred = [[id2tag[p] for p in seq] for seq in vectorized_pred]

        return iob_pred


############################################################################
#               Lowest-level (interfaces to ML modules)                #
############################################################################


def generic_train(p_or_n,
                  train_sents,
                  train_labels,
                  use_lstm,
                  val_sents=None,
                  val_labels=None,
                  test_sents=None,
                  test_labels=None,
                  dev_split=None):
    '''
    generic_train()

    Train a model that works for both prose and nonprose

    @param p_or_n.         A string that indicates "prose", "nonprose", or "all"
    @param train_sents.    A list of sentences; each sentence is tokenized
    into words
    @param train_labels.   Parallel to `train_sents`, 7-way labels for
    concept spans
    @param use_lstm        Bool indicating whether to train CRF or LSTM.
    @param val_sents.      Validation data. Same format as train_sents
    @param val_labels.     Validation data. Same format as train_labels
    @param dev_split.      A real number from 0 to 1
    '''

    # Must have data to train on:
    if len(train_sents) == 0:
        raise Exception('Training must have %s training examples' % p_or_n)

    # if you should split the data into train/dev yourself
    if (not val_sents) and (dev_split > 0.0) and (len(train_sents) > 10):

        p = int(dev_split * 100)
        sys.stdout.write('\tCreating %d/%d train/dev split\n' % (100 - p, p))

        perm = list(range(len(train_sents)))
        random.shuffle(perm)

        train_sents = [train_sents[i] for i in perm]
        train_labels = [train_labels[i] for i in perm]

        ind = int(dev_split * len(train_sents))

        val_sents = train_sents[:ind]
        train_sents = train_sents[ind:]

        val_labels = train_labels[:ind]
        train_labels = train_labels[ind:]
    else:
        sys.stdout.write('\tUsing existing validation data\n')

    sys.stdout.write('\tvectorizing words %s\n' % p_or_n)

    if use_lstm:
        print("TESTING NEW DATSET OBJECT")
        dataset = Exp.Dataset()

        parameters = hd.load_parameters_from_file("LSTM_parameters.txt")
        parameters['use_pretrained_model'] = False

        Datasets_tokens = {}
        Datasets_labels = {}
        Datasets_tokens['train'] = train_sents
        Datasets_labels['train'] = train_labels

        if val_sents:
            Datasets_tokens['valid'] = val_sents
            Datasets_labels['valid'] = val_labels

        if test_sents:
            Datasets_tokens['test'] = test_sents
            Datasets_labels['test'] = test_labels

        dataset.load_dataset(Datasets_tokens, Datasets_labels, "", parameters)
        pickle.dump(
            dataset,
            open(os.path.join(parameters['model_folder'], 'dataset.pickle'),
                 'wb'))

        print(Datasets_tokens['valid'][0])
        print(Datasets_tokens['test'][0])

        parameters['Feature_vector_length'] = dataset.feature_vector_size
        parameters['use_features_before_final_lstm'] = False
        parameters['learning_rate'] = 0.005

        sess = tf.Session()
        number_of_sent = list(range(len(dataset.token_indices['train'])))

        with sess.as_default():
            model = entity_model.EntityLSTM(dataset, parameters)
            sess.run(tf.global_variables_initializer())
            model.load_pretrained_token_embeddings(sess, dataset, parameters)
            epoch_number = -1
            transition_params_trained = np.random.rand(5 + 2, 5 + 2)
            values = {}
            values["best"] = 0

            f1_dictionary = {}
            f1_dictionary['best'] = 0

            model_saver = tf.train.Saver(max_to_keep=100)

        print("START TRAINING")

        eval_dir = os.path.join(
            tmp_dir, 'cliner_eval_%d' % random.randint(0, 256) + os.sep)
        parameters['conll_like_result_folder'] = eval_dir

        test_temp = os.path.join(parameters['conll_like_result_folder'],
                                 'test/')
        train_temp = os.path.join(parameters['conll_like_result_folder'],
                                  'train/')
        valid_temp = os.path.join(parameters['conll_like_result_folder'],
                                  'valid/')

        os.mkdir(parameters['conll_like_result_folder'])
        os.mkdir(test_temp)
        os.mkdir(train_temp)
        os.mkdir(valid_temp)

        while epoch_number < 90:
            average_loss_per_phrase = 0
            accuracy_per_phase = 0
            step = 0

            epoch_number += 1
            if epoch_number:
                sequence_numbers = list(
                    range(len(dataset.token_indices['train'])))
                random.shuffle(sequence_numbers)
                for sequence_number in sequence_numbers:
                    loss, accuracy, transition_params_trained = \
                        training_predict_LSTM.train_step(
                        sess, dataset,
                        sequence_number, model)
                    average_loss_per_phrase += loss
                    accuracy_per_phase += accuracy
                    step += 1
                    if step % 10 == 0:
                        print('Training {0:.2f}% done\n'.format(
                            step / len(sequence_numbers) * 100))

                model_saver.save(
                    sess,
                    os.path.join(parameters['model_folder'],
                                 'model_{0:05d}.ckpt'.format(epoch_number)))

                _ = average_loss_per_phrase
                _ = accuracy_per_phase

                average_loss_per_phrase = average_loss_per_phrase / len(
                    number_of_sent)
                accuracy_per_phase = accuracy_per_phase / len(number_of_sent)

            if epoch_number > 0:
                f1, _ = training_predict_LSTM.prediction_step(
                    sess, dataset, "test", model, epoch_number,
                    parameters['conll_like_result_folder'],
                    transition_params_trained)
                f1_train, _ = training_predict_LSTM.prediction_step(
                    sess, dataset, "train", model, epoch_number,
                    parameters['conll_like_result_folder'],
                    transition_params_trained)
                f1_valid, _ = training_predict_LSTM.prediction_step(
                    sess, dataset, "valid", model, epoch_number,
                    parameters['conll_like_result_folder'],
                    transition_params_trained)

                correctly_predicted_tokens = \
                    training_predict_LSTM.compute_train_accuracy(
                    parameters[
                        'conll_like_result_folder'] + "valid" + os.sep +
                    "epoche_" + str(
                        epoch_number) + ".txt")

                if f1_dictionary['best'] < float(f1_valid):
                    f1_dictionary['epoche'] = epoch_number
                    f1_dictionary['best'] = float(f1_valid)

                if values["best"] < correctly_predicted_tokens:
                    values["epoche"] = epoch_number
                    values["best"] = correctly_predicted_tokens

                # print ("Number of correctly predicted tokens -test "+str(
                # correctly_predicted_tokens))

                print("NEW EPOCHE" + " " + str(epoch_number))

                print("Current F1 on train" + " " + str(f1_train))
                print("Current F1 on valid" + " " + str(f1_valid))
                print("Current F1 on test" + " " + str(f1))

                print("Current F1 best (validation): ")
                print(f1_dictionary)

        shutil.rmtree(parameters['conll_like_result_folder'])
        return parameters, dataset, f1_dictionary['best']

    else:
        ########
        # CRF
        ########

        # vectorize tokenized sentences
        text_features = extract_features(train_sents)
        # type(text_features): <type 'list'>

        # Collect list of feature types
        enabled_features = set()
        for sf in text_features:
            for wf in sf:
                for (feature_type, _), _ in wf.items():
                    if feature_type.startswith('prev'):
                        feature_type = 'PREV*'
                    if feature_type.startswith('next'):
                        feature_type = 'NEXT*'
                    enabled_features.add(feature_type)
        enabled_features = sorted(enabled_features)

        # Vectorize features
        vocab = DictVectorizer()
        flat_X_feats = vocab.fit_transform(flatten(text_features))
        X_feats = reconstruct_list(flat_X_feats,
                                   save_list_structure(text_features))

        # vectorize IOB labels
        Y_labels = [[tag2id[y] for y in y_seq] for y_seq in train_labels]

        assert len(X_feats) == len(Y_labels)
        for _, i in enumerate(range(len(X_feats))):
            assert X_feats[i].shape[0] == len(Y_labels[i])

        # if there is specified validation data, then vectorize it
        if val_sents:
            # vectorize validation X
            val_text_features = extract_features(val_sents)
            flat_val_X_feats = vocab.transform(flatten(val_text_features))
            val_X = reconstruct_list(flat_val_X_feats,
                                     save_list_structure(val_text_features))
            # vectorize validation Y
            val_Y = [[tag2id[y] for y in y_seq] for y_seq in val_labels]

        # if there is specified test data, then vectorize it
        if test_sents:
            # vectorize test X
            test_text_features = extract_features(test_sents)
            flat_test_X_feats = vocab.transform(flatten(test_text_features))
            test_X = reconstruct_list(flat_test_X_feats,
                                      save_list_structure(test_text_features))
            # vectorize test Y
            test_Y = [[tag2id[y] for y in y_seq] for y_seq in test_labels]
        else:
            test_X = None
            test_Y = None

    sys.stdout.write('\ttraining classifiers %s\n' % p_or_n)

    if use_lstm:
        # train using lstm
        # clf, dev_score  = keras_ml.train(X_seq_ids, Y_labels, tag2id,
        # len(vocab),
        #                                  val_X_ids=val_X, val_Y_ids=val_Y,
        #                                  test_X_ids=test_X, test_Y_ids=test_Y)
        clf, dev_score = None, None
    else:
        # train using crf
        clf, dev_score = crf.train(X_feats,
                                   Y_labels,
                                   val_X=val_X,
                                   val_Y=val_Y,
                                   test_X=test_X,
                                   test_Y=test_Y)

    return vocab, clf, dev_score, enabled_features


# def generic_predict(p_or_n, tokenized_sents, vocab, clf, use_lstm,
# pretrained_dataset=None,tokens_to_vec=None,  current_model=None,
# parameters=None):
def generic_predict(p_or_n, tokenized_sents, vocab, clf, use_lstm):
    '''
    generic_predict()

    Train a model that works for both prose and nonprose

    @param p_or_n.          A string that indicates "prose", "nonprose",
    or "all"
    @param tokenized_sents. A list of sentences, where each sentence is
    tokenized
                              into words
    @param vocab.           A dictionary mapping word tokens to numeric indices.
    @param clf.             An encoding of the trained keras model.
    @param use_lstm.        Bool indicating whether clf is a CRF or LSTM.
    '''
    # use_lstm=self._use_lstm
    if use_lstm:

        parameters = hd.load_parameters_from_file("LSTM_parameters.txt")
        parameters['use_pretrained_model'] = True

        # model_folder="./models/NN_models"
        predictions = []
        sys.stdout.write('\n use_lstm \n')
        dataset = Exp.Dataset()

        fictional_labels = copy.deepcopy(tokenized_sents)
        for idx, x in enumerate(fictional_labels):
            for val_id, _ in enumerate(x):
                fictional_labels[idx][val_id] = 'O'

        Datasets_tokens = {}
        Datasets_labels = {}

        Datasets_tokens['deploy'] = tokenized_sents
        Datasets_labels['deploy'] = fictional_labels

        token_to_vector = dataset.load_dataset(Datasets_tokens,
                                               Datasets_labels,
                                               "",
                                               parameters,
                                               token_to_vector=None,
                                               pretrained_dataset=None)

        print(dataset.token_indices.keys())

        parameters['Feature_vector_length'] = dataset.feature_vector_size
        parameters['use_features_before_final_lstm'] = False

        dataset.update_dataset("", ['deploy'], Datasets_tokens,
                               Datasets_labels)

        del Datasets_tokens
        del Datasets_labels

        # model=current_model
        model = entity_model.EntityLSTM(dataset, parameters)

        os.mkdir(parameters['conll_like_result_folder'])

        test_temp = os.path.join(parameters['conll_like_result_folder'],
                                 'test/')
        train_temp = os.path.join(parameters['conll_like_result_folder'],
                                  'train/')
        valid_temp = os.path.join(parameters['conll_like_result_folder'],
                                  'valid/')

        os.mkdir(test_temp)
        os.mkdir(train_temp)
        os.mkdir(valid_temp)

        sess = tf.Session()
        with sess.as_default():

            # model=entity_model.EntityLSTM(dataset,parameters)
            transition_params_trained = model.restore_from_pretrained_model(
                parameters,
                dataset,
                sess,
                token_to_vector=token_to_vector,
                pretrained_dataset=None)
            del token_to_vector
            predictions = training_predict_LSTM.prediction_step(
                sess, dataset, "deploy", model, 0,
                parameters['conll_like_result_folder'],
                transition_params_trained)
            sess.close()

        tf.reset_default_graph()

        shutil.rmtree(parameters['conll_like_result_folder'])
        return predictions, model

    # If nothing to predict, skip actual prediction
    if len(tokenized_sents) == 0:
        sys.stdout.write('\tnothing to predict %s\n' % p_or_n)
        return []

    sys.stdout.write('\tvectorizing words %s\n' % p_or_n)

    if use_lstm:
        print('todo: incorporate lstm')
        # vectorize tokenized sentences
        # X = []
        # for sent in tokenized_sents:
        #   id_seq = []
        #   for w in sent:
        #      if w in vocab:
        #           id_seq.append(vocab[w])
        #       else:
        #        id_seq.append(vocab['oov'])
        #  X.append(id_seq)
    else:

        # vectorize validation X
        text_features = extract_features(tokenized_sents)
        flat_X_feats = vocab.transform(flatten(text_features))
        X = reconstruct_list(flat_X_feats, save_list_structure(text_features))

    sys.stdout.write('\tpredicting  labels %s\n' % p_or_n)

    # Predict labels
    if use_lstm:
        print("TEST_PREDICT")
        sys.exit()

    else:
        predictions = crf.predict(clf, X)

    # Format labels from output
    return predictions
