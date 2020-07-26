import tensorflow as tf
import numpy as np
import codecs
import re
import time
#import utils_tf
#import utils_nlp
import helper_dataset as hd
import tensorflow.contrib.layers as layers
import os
import pickle
import utils_tf

# TO DO: ADD CNN LAYER

def bidirectional_GRU(input,hidden_state_dimension,initializer,sequence_length=None, output_sequence=True):
    print ("Biderectional GRU")
    with tf.variable_scope("biderectional_GRU"):
        if sequence_length==None:
            batch_size=1 # ONE WORD(char)
            sequence_length = tf.shape(input)[1]
            sequence_length = tf.expand_dims(sequence_length, axis=0, name='sequence_length')  #NOT SURE IF IT EVER HAPPENS
        else:
            batch_size= tf.shape(sequence_length)[0]
            
            
        gru_cell={}
        initial_state={}
        for direction in ["forward","backward"]: 
            gru_cell[direction] = tf.contrib.rnn.GRUCell(hidden_state_dimension)  
            initial_state[direction]=gru_cell[direction].zero_state(batch_size, tf.float32)           
        outputs,final_states = tf.nn.bidirectional_dynamic_rnn(gru_cell["forward"],gru_cell["backward"],input, sequence_length=sequence_length,initial_state_fw=initial_state["forward"],initial_state_bw=initial_state["backward"])    

        
        if output_sequence==True:
           outputs_forward, outputs_backward = outputs        
           output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')

        else:
            final_states_forward, final_states_backward = final_states                      

            output = tf.concat([final_states_forward, final_states_backward], axis=1, name='output') #111

        return output




def bidirectional_LSTM(input, hidden_state_dimension, initializer, sequence_length=None, output_sequence=True):
    
    print ("Biderectional LSTM")
    with tf.variable_scope("bidirectional_LSTM"):
        if sequence_length == None:
            batch_size = 1
            sequence_length = tf.shape(input)[1]
            sequence_length = tf.expand_dims(sequence_length, axis=0, name='sequence_length')
        else:
            batch_size = tf.shape(sequence_length)[0]

        lstm_cell = {}
        initial_state = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                # LSTM cell
                lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(hidden_state_dimension, use_peepholes=False, forget_bias=1.0, initializer=initializer, state_is_tuple=True, activation=tf.tanh) # tf.tanh (default to RELU)
               # lstm_cell[direction] = tf.contrib.rnn_cell.GRUCell(hidden_state_dimension,activation=tf.tanh,)
                
                
                # initial state: http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
                initial_cell_state = tf.get_variable("initial_cell_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
                initial_output_state = tf.get_variable("initial_output_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
                c_states = tf.tile(initial_cell_state, tf.stack([batch_size, 1]))
                h_states = tf.tile(initial_output_state, tf.stack([batch_size, 1]))
                initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

        # sequence_length must be provided for tf.nn.bidirectional_dynamic_rnn due to internal bug
        outputs, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                                                    lstm_cell["backward"],
                                                                    input,
                                                                    dtype=tf.float32,
                                                                    sequence_length=sequence_length,
                                                                    initial_state_fw=initial_state["forward"],
                                                                    initial_state_bw=initial_state["backward"])
        if output_sequence == True:
            outputs_forward, outputs_backward = outputs
            output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
        else:
            # max pooling
#             outputs_forward, outputs_backward = outputs
#             output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')
#             output = tf.reduce_max(output, axis=1, name='output')
            # last pooling
            final_states_forward, final_states_backward = final_states
            output = tf.concat([final_states_forward[1], final_states_backward[1]], axis=1, name='output')

    return output




class EntityLSTM(object):
    """
    An LSTM architecture for named entity recognition.
    Uses a character embedding layer followed by an LSTM to generate vector representation from characters for each token.
    Then the character vector is concatenated with token embedding vector, which is input to another LSTM  followed by a CRF layer.
    """
    def __init__(self, dataset, parameters):

        self.verbose = False
        self.feature_vector_length=parameters['Feature_vector_length']

        # Placeholders for input, output and dropout
        self.input_token_indices = tf.placeholder(tf.int32, [None], name="input_token_indices")
        self.input_label_indices_vector = tf.placeholder(tf.float32, [None, dataset.number_of_classes], name="input_label_indices_vector")
        self.input_label_indices_flat = tf.placeholder(tf.int32, [None], name="input_label_indices_flat")
        self.input_token_character_indices = tf.placeholder(tf.int32, [None, None], name="input_token_indices")
        self.input_token_lengths = tf.placeholder(tf.int32, [None], name="input_token_lengths")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        self.input_features=tf.placeholder(tf.float32, [None,self.feature_vector_length], name="features")
        
        
        self.vocabulary_size=dataset.vocabulary_size

        # Internal parameters
        initializer = tf.contrib.layers.xavier_initializer()

        if parameters['use_character_lstm']:
            with tf.variable_scope("character_embedding"):
                self.character_embedding_weights = tf.get_variable(
                    "character_embedding_weights",
                    shape=[dataset.alphabet_size, parameters['character_embedding_dimension']],
                    initializer=initializer)
                embedded_characters = tf.nn.embedding_lookup(self.character_embedding_weights, self.input_token_character_indices, name='embedded_characters')
                if self.verbose: print("embedded_characters: {0}".format(embedded_characters))
               # utils_tf.variable_summaries(self.character_embedding_weights)

            # Character LSTM layer
            with tf.variable_scope('character_lstm') as vs:
              if parameters['Use_LSTM']==True:
                character_lstm_output = bidirectional_LSTM(embedded_characters, parameters['character_lstm_hidden_state_dimension'], initializer,
                                                           sequence_length=self.input_token_lengths, output_sequence=False)
                self.character_lstm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
              else:
                 character_lstm_output = bidirectional_GRU(embedded_characters, parameters['character_lstm_hidden_state_dimension'], initializer,
                                                           sequence_length=self.input_token_lengths, output_sequence=False) 
            # Attention, not implemented      

            #  with tf.variable_scope('attention') as scope:
             #    word_level_output = task_specific_attention(character_lstm_output,dataset.token_lengths,scope=scope)
                # print (w)

             # sentence_inputs = tf.reshape(word_level_output, [self.document_size, self.sentence_size, self.word_output_size])





        # Token embedding layer
        with tf.variable_scope("token_embedding"):
            self.token_embedding_weights = tf.get_variable(
                "token_embedding_weights",
                shape=[dataset.vocabulary_size, parameters['token_embedding_dimension']],
                initializer=initializer,
                trainable=not parameters['freeze_token_embeddings'])
            embedded_tokens = tf.nn.embedding_lookup(self.token_embedding_weights, self.input_token_indices)
          #  utils_tf.variable_summaries(self.token_embedding_weights)

        # Concatenate character LSTM outputs and token embeddings
        if parameters['use_character_lstm']:
            with tf.variable_scope("concatenate_token_and_character_vectors"):
                if self.verbose: print('embedded_tokens: {0}'.format(embedded_tokens))
                token_lstm_input = tf.concat([character_lstm_output, embedded_tokens], axis=1, name='token_lstm_input')
                if self.verbose: print("token_lstm_input: {0}".format(token_lstm_input))
        else:
            token_lstm_input = embedded_tokens
            
        if parameters['use_features_before_final_lstm']:
            with tf.variable_scope("features_argumentation_pre_LSTM"):
                token_lstm_input=tf.concat([token_lstm_input, self.input_features], 1)
                print (token_lstm_input)
            

        # Add dropout
        with tf.variable_scope("dropout"):
            token_lstm_input_drop = tf.nn.dropout(token_lstm_input, self.dropout_keep_prob, name='token_lstm_input_drop')
            if self.verbose: print("token_lstm_input_drop: {0}".format(token_lstm_input_drop))
            # https://www.tensorflow.org/api_guides/python/contrib.rnn
            # Prepare data shape to match `rnn` function requirements
            # Current data input shape: (batch_size, n_steps, n_input)
            # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
            token_lstm_input_drop_expanded = tf.expand_dims(token_lstm_input_drop, axis=0, name='token_lstm_input_drop_expanded')
            if self.verbose: print("token_lstm_input_drop_expanded: {0}".format(token_lstm_input_drop_expanded))
            
        #if parameters['use_features_before_final_lstm']:
        #   with tf.variable_scope("features_argumentation_pre_LSTM"):
        #       token_lstm_input_drop_expanded=tf.concat([token_lstm_input_drop_expanded, self.input_features], 1)
        #       print (token_lstm_input_drop_expanded)

        # Token LSTM layer
        with tf.variable_scope('token_lstm') as vs:
            if parameters['Use_LSTM']==True: token_lstm_output = bidirectional_LSTM(token_lstm_input_drop_expanded, parameters['token_lstm_hidden_state_dimension'], initializer, output_sequence=True)
            else: token_lstm_output = bidirectional_GRU(token_lstm_input_drop_expanded, parameters['token_lstm_hidden_state_dimension'], initializer, output_sequence=True)
            token_lstm_output_squeezed = tf.squeeze(token_lstm_output, axis=0)
            self.token_lstm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # Needed only if Bidirectional LSTM is used for token level
        with tf.variable_scope("feedforward_after_lstm") as vs:
            W = tf.get_variable(
                "W",
                shape=[2 * parameters['token_lstm_hidden_state_dimension'], parameters['token_lstm_hidden_state_dimension']],
                initializer=initializer)
            b = tf.Variable(tf.constant(0.0, shape=[parameters['token_lstm_hidden_state_dimension']]), name="bias")
            outputs = tf.nn.xw_plus_b(token_lstm_output_squeezed, W, b, name="output_before_tanh")
            outputs = tf.nn.tanh(outputs, name="output_after_tanh")
            self.token_lstm_variables += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        with tf.variable_scope("feedforward_before_crf") as vs:
            W = tf.get_variable(
                "W",
                shape=[parameters['token_lstm_hidden_state_dimension'], dataset.number_of_classes],
                initializer=initializer)
            b = tf.Variable(tf.constant(0.0, shape=[dataset.number_of_classes]), name="bias")
            scores = tf.nn.xw_plus_b(outputs, W, b, name="scores")
            self.unary_scores = scores
            self.predictions = tf.argmax(self.unary_scores, 1, name="predictions")
            #utils_tf.variable_summaries(W)
           # utils_tf.variable_summaries(b)
            self.feedforward_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # CRF layer
        if parameters['use_crf']:
            print ("CRF IS IN USE")
            with tf.variable_scope("crf") as vs:
                # Add start and end tokens
                small_score = -1000.0
                large_score = 0.0
                sequence_length = tf.shape(self.unary_scores)[0]
                unary_scores_with_start_and_end = tf.concat([self.unary_scores, tf.tile( tf.constant(small_score, shape=[1, 2]) , [sequence_length, 1])], 1)
                start_unary_scores = [[small_score] * dataset.number_of_classes + [large_score, small_score]]
                end_unary_scores = [[small_score] * dataset.number_of_classes + [small_score, large_score]]
                self.unary_scores = tf.concat([start_unary_scores, unary_scores_with_start_and_end, end_unary_scores], 0)
                start_index = dataset.number_of_classes
                end_index = dataset.number_of_classes + 1
                input_label_indices_flat_with_start_and_end = tf.concat([ tf.constant(start_index, shape=[1]), self.input_label_indices_flat, tf.constant(end_index, shape=[1]) ], 0)

                # Apply CRF layer
                sequence_length = tf.shape(self.unary_scores)[0]
                sequence_lengths = tf.expand_dims(sequence_length, axis=0, name='sequence_lengths')
                unary_scores_expanded = tf.expand_dims(self.unary_scores, axis=0, name='unary_scores_expanded')
                input_label_indices_flat_batch = tf.expand_dims(input_label_indices_flat_with_start_and_end, axis=0, name='input_label_indices_flat_batch')
                if self.verbose: print('unary_scores_expanded: {0}'.format(unary_scores_expanded))
                if self.verbose: print('input_label_indices_flat_batch: {0}'.format(input_label_indices_flat_batch))
                if self.verbose: print("sequence_lengths: {0}".format(sequence_lengths))
                # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/crf
                # Compute the log-likelihood of the gold sequences and keep the transition params for inference at test time.
                self.transition_parameters=tf.get_variable(
                    "transitions",
                    shape=[dataset.number_of_classes+2, dataset.number_of_classes+2],
                    initializer=initializer)
                #utils_tf.variable_summaries(self.transition_parameters)
                log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                    unary_scores_expanded, input_label_indices_flat_batch, sequence_lengths, transition_params=self.transition_parameters)
                self.loss =  tf.reduce_mean(-log_likelihood, name='cross_entropy_mean_loss')
                self.accuracy = tf.constant(1)

                self.crf_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name) # LATER FOR RESTORE

        # Do not use CRF layer
        else:
            with tf.variable_scope("crf") as vs:
                self.transition_parameters = tf.get_variable(
                    "transitions",
                    shape=[dataset.number_of_classes+2, dataset.number_of_classes+2],
                    initializer=initializer)
               # utils_tf.variable_summaries(self.transition_parameters)
                self.crf_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

            # Calculate mean cross-entropy loss
            with tf.variable_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.unary_scores, labels=self.input_label_indices_vector, name='softmax')
                self.loss =  tf.reduce_mean(losses, name='cross_entropy_mean_loss')
            with tf.variable_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label_indices_vector, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')

        self.define_training_procedure(parameters)
        self.summary_op = tf.summary.merge_all()
        self.saver = tf.train.Saver(max_to_keep=100)
        
     

    def define_training_procedure(self, parameters):
        # Define training procedure
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        if parameters['optimizer'] == 'adam':
            self.optimizer = tf.train.AdamOptimizer(parameters['learning_rate'])
        elif parameters['optimizer'] == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(parameters['learning_rate'])
        elif parameters['optimizer'] == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(parameters['learning_rate'])
        else:
            raise ValueError('The lr_method parameter must be either adadelta, adam or sgd.')

        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        #MODIFY:
        if parameters['gradient_clipping_value']:
         def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -5.0, 5.0)
         grads_and_vars = [(ClipIfNotNone(grad), var) for grad, var in grads_and_vars]
        
        self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    
    def load_pretrained_token_embeddings(self, sess, dataset, parameters, token_to_vector=None):
        if parameters['token_pretrained_embedding_filepath'] == '':
            return
        # Load embeddings
        start_time = time.time()
        print('Load token embeddings... ', end='', flush=True)
        if token_to_vector == None:
            token_to_vector = hd.load_pretrained_token_embeddings(parameters)

        initial_weights = sess.run(self.token_embedding_weights.read_value())
        number_of_loaded_word_vectors = 0
        number_of_token_original_case_found = 0
        number_of_token_lowercase_found = 0
        number_of_token_digits_replaced_with_zeros_found = 0
        number_of_token_lowercase_and_digits_replaced_with_zeros_found = 0
        for token in dataset.token_to_index.keys():
            if token in token_to_vector.keys():
                initial_weights[dataset.token_to_index[token]] = token_to_vector[token]
                number_of_token_original_case_found += 1
            elif parameters['check_for_lowercase'] and token.lower() in token_to_vector.keys():
                initial_weights[dataset.token_to_index[token]] = token_to_vector[token.lower()]
                number_of_token_lowercase_found += 1
            elif parameters['check_for_digits_replaced_with_zeros'] and re.sub('\d', '0', token) in token_to_vector.keys():
                initial_weights[dataset.token_to_index[token]] = token_to_vector[re.sub('\d', '0', token)]
                number_of_token_digits_replaced_with_zeros_found += 1
            elif parameters['check_for_lowercase'] and parameters['check_for_digits_replaced_with_zeros'] and re.sub('\d', '0', token.lower()) in token_to_vector.keys():
                initial_weights[dataset.token_to_index[token]] = token_to_vector[re.sub('\d', '0', token.lower())]
                number_of_token_lowercase_and_digits_replaced_with_zeros_found += 1
            else:
                continue
            number_of_loaded_word_vectors += 1
        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))
        print("number_of_token_original_case_found: {0}".format(number_of_token_original_case_found))
        print("number_of_token_lowercase_found: {0}".format(number_of_token_lowercase_found))
        print("number_of_token_digits_replaced_with_zeros_found: {0}".format(number_of_token_digits_replaced_with_zeros_found))
        print("number_of_token_lowercase_and_digits_replaced_with_zeros_found: {0}".format(number_of_token_lowercase_and_digits_replaced_with_zeros_found))
        print('number_of_loaded_word_vectors: {0}'.format(number_of_loaded_word_vectors))
        print("dataset.vocabulary_size: {0}".format(dataset.vocabulary_size))
        sess.run(self.token_embedding_weights.assign(initial_weights))
   

    def load_embeddings_from_pretrained_model(self, sess, dataset, pretraining_dataset, pretrained_embedding_weights, embedding_type='token'):
        if embedding_type == 'token':
            embedding_weights = self.token_embedding_weights
            index_to_string = dataset.index_to_token
            pretraining_string_to_index = pretraining_dataset.token_to_index
        elif embedding_type == 'character':
            embedding_weights = self.character_embedding_weights
            index_to_string = dataset.index_to_character
            pretraining_string_to_index = pretraining_dataset.character_to_index
        # Load embeddings
        start_time = time.time()
        print('Load {0} embeddings from pretrained model... '.format(embedding_type), end='', flush=True)
        initial_weights = sess.run(embedding_weights.read_value())

        if embedding_type == 'token':
            initial_weights[dataset.UNK_TOKEN_INDEX] = pretrained_embedding_weights[pretraining_dataset.UNK_TOKEN_INDEX]
        elif embedding_type == 'character':
            initial_weights[dataset.PADDING_CHARACTER_INDEX] = pretrained_embedding_weights[pretraining_dataset.PADDING_CHARACTER_INDEX]

        number_of_loaded_vectors = 1
        for index, string in index_to_string.items():
            if index == dataset.UNK_TOKEN_INDEX:
                continue
            if string in pretraining_string_to_index.keys():
                initial_weights[index] = pretrained_embedding_weights[pretraining_string_to_index[string]]
                number_of_loaded_vectors += 1
        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))
        print("number_of_loaded_vectors: {0}".format(number_of_loaded_vectors))
        if embedding_type == 'token':
            print("dataset.vocabulary_size: {0}".format(dataset.vocabulary_size))
        elif embedding_type == 'character':
            print("dataset.alphabet_size: {0}".format(dataset.alphabet_size))
        sess.run(embedding_weights.assign(initial_weights))


    def resize_without_redoing_model(self, parameters,new_dataset_vocab_size,sess):
        ""
        utils_tf.resize_tensor_variable(sess, self.token_embedding_weights, [new_dataset_vocab_size, parameters['token_embedding_dimension']])
         


    def restore_from_pretrained_model(self, parameters, dataset, sess, token_to_vector=None,pretrained_dataset=None):

        temp_pretrained_dataset_adress=parameters['model_folder']+os.sep+"dataset.pickle"
        temp_pretrained_model_adress=parameters['model_folder']+os.sep+parameters['model_name']
        
        print (temp_pretrained_model_adress)
        
        if pretrained_dataset==None:
            pretraining_dataset = pickle.load(open(temp_pretrained_dataset_adress, 'rb')) 
        else:
            print ("PRETRAINING HERE")
            pretraining_dataset=pretrained_dataset 
            
            
        pretrained_model_checkpoint_filepath = temp_pretrained_model_adress
        
        assert pretraining_dataset.index_to_label == dataset.index_to_label  # DEBUG fron  F&J
    
        # If the token and character mappings are exactly the same
        if pretraining_dataset.index_to_token == dataset.index_to_token and pretraining_dataset.index_to_character == dataset.index_to_character:
            
            # Restore the pretrained model
            self.saver.restore(sess, pretrained_model_checkpoint_filepath) # Works only when the dimensions of tensor variables are matched.
            del  pretraining_dataset
        
        # If the token and character mappings are different between the pretrained model and the current model
        else:
            print ("INDEX TO TOKEN DO NOT MATCH")
            
            # Resize the token and character embedding weights to match them with the pretrained model (required in order to restore the pretrained model)
            utils_tf.resize_tensor_variable(sess, self.character_embedding_weights, [pretraining_dataset.alphabet_size, parameters['character_embedding_dimension']])
            utils_tf.resize_tensor_variable(sess, self.token_embedding_weights, [pretraining_dataset.vocabulary_size, parameters['token_embedding_dimension']])
        
            # Restore the pretrained model
            self.saver.restore(sess, pretrained_model_checkpoint_filepath) # Works only when the dimensions of tensor variables are matched.
            
            # Get pretrained embeddings
            character_embedding_weights, token_embedding_weights = sess.run([self.character_embedding_weights, self.token_embedding_weights]) 
            
            # Restore the sizes of token and character embedding weights
            utils_tf.resize_tensor_variable(sess, self.character_embedding_weights, [dataset.alphabet_size, parameters['character_embedding_dimension']])
            utils_tf.resize_tensor_variable(sess, self.token_embedding_weights, [dataset.vocabulary_size, parameters['token_embedding_dimension']]) 
            
            # Re-initialize the token and character embedding weights
            sess.run(tf.variables_initializer([self.character_embedding_weights, self.token_embedding_weights]))
            
            # Load embedding weights from pretrained token embeddings first
            self.load_pretrained_token_embeddings(sess, dataset, parameters, token_to_vector=token_to_vector) 
            self.load_embeddings_from_pretrained_model(sess, dataset, pretraining_dataset, token_embedding_weights, embedding_type='token')
            self.load_embeddings_from_pretrained_model(sess, dataset, pretraining_dataset, character_embedding_weights, embedding_type='character') 
            
            del pretraining_dataset
            del character_embedding_weights
            del token_embedding_weights
        
        # Get transition parameters
        transition_params_trained = sess.run(self.transition_parameters)
        
        parameters={'reload_character_embeddings': True, 'reload_character_lstm':True, 'reload_token_embeddings':True, 'reload_token_lstm':True, 'reload_feedforward':True, 'reload_crf':True}
        if not parameters['reload_character_embeddings']:
            sess.run(tf.variables_initializer([self.character_embedding_weights]))
        if not parameters['reload_character_lstm']:
            sess.run(tf.variables_initializer(self.character_lstm_variables))
        if not parameters['reload_token_embeddings']:
            sess.run(tf.variables_initializer([self.token_embedding_weights]))
        if not parameters['reload_token_lstm']:
            sess.run(tf.variables_initializer(self.token_lstm_variables))
        if not parameters['reload_feedforward']:
            sess.run(tf.variables_initializer(self.feedforward_variables))
        if not parameters['reload_crf']:
            sess.run(tf.variables_initializer(self.crf_variables))
    
    
       
        return transition_params_trained
    
    














