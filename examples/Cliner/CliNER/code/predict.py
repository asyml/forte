######################################################################
#  CliNER - predict.py                                               #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Use trained model to predict concept labels for data.    #
######################################################################

import os
import sys
import glob
import argparse
import itertools
import pickle

from examples.Cliner.CliNER.code import tools
from examples.Cliner.CliNER.code.model import ClinerModel, write
from examples.Cliner.CliNER.code.notes.documents import Document
import copy
import sys,os
import examples.Cliner.CliNER.code.model

class CliNERPredict():
    def __init__(self, txt, output, model_path, format):

        self.txt = txt
        self.output = output
        self.model_path = model_path
        self.format = format

        self.files = glob.glob(self.txt)
        tools.mkpath(self.output)

        if self.format:
            pass
        else:
            parser.print_help(sys.stderr)
            sys.stderr.write('\n\tERROR: must provide "format" argument\n\n')
            exit(1)

    # Predict
    def predict(self):
        # Must specify output format
        if self.format not in ['i2b2']:
            sys.stderr.write('\n\tError: Must specify output format\n')
            sys.stderr.write('\tAvailable formats: i2b2\n')
            sys.stderr.write('\n')
            exit(1)

        # Load model
        #if use_lstm==False:
        with open(self.model_path, 'rb') as f:
            Model = pickle.load(f, encoding='latin1')

        if Model._use_lstm:
            import helper_dataset as hd
            import DatasetCliner_experimental as Exp
            import entity_lstm as entity_model

            parameters=hd.load_parameters_from_file("LSTM_parameters.txt")
            parameters['use_pretrained_model']=True

            temp_pretrained_dataset_adress=parameters['model_folder']+os.sep+"dataset.pickle"
            Model._pretrained_dataset = pickle.load(open(temp_pretrained_dataset_adress, 'rb'))
            Model._pretrained_wordvector=hd.load_pretrained_token_embeddings(parameters)
            Model._current_model=None

            print ("END TEST")
            #exit()
            #model.parameters=None

        # Tell user if not predicting
        if not self.files:
            sys.stderr.write("\n\tNote: You did not supply any input files\n\n")
            exit()

        n = len(self.files)

        for i,txt in enumerate(sorted(self.files)):
            note = Document(txt)

            # Output file
            fname = os.path.splitext(os.path.basename(txt))[0] + '.' + 'con'
            out_path = os.path.join(self.output, fname)

            #'''
            if os.path.exists(out_path):
                print('\tWARNING: prediction file already exists (%s)' % out_path)
                #continue
            #'''

            sys.stdout.write('%s\n' % ('-' * 30))
            sys.stdout.write('\n\t%d of %d\n' % (i+1,n))
            sys.stdout.write('\t%s\n\n' % txt)

            # Predict concept labels
            labels = Model.predict_classes_from_document(note)

            # Get predictions in proper format
            output = note.write(labels)

            # Output the concept predictions
            sys.stdout.write('\n\nwriting to: %s\n' % out_path)
            with open(out_path, 'w') as f:
                f.write('%s\n' % output)
            sys.stdout.write('\n')



if __name__ == '__main__':
    Model = CliNERPredict()
    Model.predict()

