######################################################################
#  CliNER - train.py                                                 #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Build model for given training data.                     #
######################################################################


import glob
import os.path
import pickle
import sys

from examples.Cliner.CliNER.code import tools
from examples.Cliner.CliNER.code.model import ClinerModel
from examples.Cliner.CliNER.code.notes.documents import Document

# base directory
CLINER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CliNERTrain():
    def __init__(self, txt, con, output, model_path, format):

        self.txt = txt
        self.con = con
        self.output = output
        self.model_path = model_path
        self.format = format

        self.val_txt = None
        self.val_con = None
        self.test_txt = None
        self.test_con = None
        self.use_lstm = False

        # A list of txt and concept file paths
        train_txt_files = glob.glob(self.txt)
        train_con_files = glob.glob(self.con)

        # data format
        if self.format:
            format = self.format

        # Must specify output format
        if self.format not in ['i2b2']:
            sys.stderr.write('\n')
            sys.exit()

        # Collect training data file paths
        train_txt_files_map = tools.map_files(train_txt_files)
        train_con_files_map = tools.map_files(train_con_files)

        self.training_list = []
        for k in train_txt_files_map:
            if k in train_con_files_map:
                self.training_list.append(
                    (train_txt_files_map[k], train_con_files_map[k]))

        # If validation data was specified
        if self.val_txt and self.val_con:
            val_txt_files = glob.glob(self.val_txt)
            val_con_files = glob.glob(self.val_con)

            val_txt_files_map = tools.map_files(val_txt_files)
            val_con_files_map = tools.map_files(val_con_files)

            val_list = []
            for k in val_txt_files_map:
                if k in val_con_files_map:
                    val_list.append(
                        (val_txt_files_map[k], val_con_files_map[k]))
        else:
            val_list = []

        # If test data was specified
        if self.test_txt and self.test_con:
            test_txt_files = glob.glob(self.test_txt)
            test_con_files = glob.glob(self.test_con)

            test_txt_files_map = tools.map_files(test_txt_files)
            test_con_files_map = tools.map_files(test_con_files)

            test_list = []
            for k in test_txt_files_map:
                if k in test_con_files_map:
                    test_list.append(
                        (test_txt_files_map[k], test_con_files_map[k]))
        else:
            test_list = []

    def train(self):
        val = []
        test = []
        logfile = 'CliNER/models/train.log'

        # Read the data into a Document object
        train_docs = []
        for txt, con in self.training_list:
            doc_tmp = Document(txt, con)
            train_docs.append(doc_tmp)

        val_docs = []
        for txt, con in val:
            doc_tmp = Document(txt, con)
            val_docs.append(doc_tmp)

        test_docs = []
        for txt, con in test:
            doc_tmp = Document(txt, con)
            test_docs.append(doc_tmp)

        # file names
        if not train_docs:
            print('Error: Cannot train on 0 files. Terminating train.')
            return 1

        # Create a Machine Learning model
        model = ClinerModel(self.use_lstm)

        # Train the model using the Documents's data
        model.train(train_docs, val=val_docs, test=test_docs)

        # Pickle dump
        print('\nserializing model to %s\n' % self.model_path)
        with open(self.model_path, "wb") as m_file:
            pickle.dump(model, m_file)

        model.log(logfile, model_file=self.model_path)
        model.log(sys.stdout, model_file=self.model_path)
