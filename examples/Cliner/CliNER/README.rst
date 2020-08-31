===============================
CliNER
===============================

Clinical Named Entity Recognition system (CliNER) is an open-source natural language processing system for named entity recognition in clinical text of electronic health records. CliNER system is designed to follow best practices in clinical concept extraction, as established in i2b2 2010 shared task.

CliNER is implemented as a sequence classification task, where every token is predicted IOB-style as either: Problem, Test, Treatment, or None. Command line flags let you specify two different sequence classification algorithms:
    1. CRF (default) - with linguistic and domain-specific features
    2. LSTM

Please note that for optimal performance, CliNER requires the users to obtain a Unified Medical Language System (UMLS) license, since UMLS Metathesaurus is used as one of the knowledge sources for the above classifiers.


* Free software: Apache v2.0 license
* See the CliNER Wiki page for additional resources. 
  
  https://github.com/text-machine-lab/CliNER/wiki
  

Out-of-the-Box Model
--------

Although i2b2 licensing prevents us from releasing our cliner models trained on i2b2 data, we generated some comparable models from automatically-annotated MIMIC II text.

This silver MIMIC model can be found at http://text-machine.cs.uml.edu/cliner/models/silver.crf

Installation
--------

        $ git clone https://github.com/text-machine-lab/CliNER.git

        $ pip install -r requirements.txt
        
Verifying Installation
--------
        
        $ wget http://text-machine.cs.uml.edu/cliner/models/silver.crf
        
        $ mv silver.crf models/silver.crf
        
        $ cliner predict --txt data/examples/ex_doc.txt --out data/predictions --model models/silver.crf --format i2b2

If you *do not* run into build errors, then your installation is complete.

Example Data
--------

Although we cannot provide i2b2 data, there is a sample to demonstrate how the data is formatted (not actual data from i2b2, though).

    data/examples/ex_doc.txt

This is a text file. Discharge summaries are written out in plaintext, just like this. It is paired with a concept file, which has its annotations.

    data/examples/ex_doc.con

This is a concept file. It provides annotations for the concepts (problems, treatments, and tests) of the text file. The format is as follows - each instance of a concept has one line. The line shows the text span, the line number, token numbers of the span (delimited by white space), and the label of the concept.

**Please note that the example data is simply one of many examples that can found online.**

Usage
--------

Here are some use cases:

(1) Help

This help message will list the options available to run (train/predict/evaluate)

    cliner --help

(2) Training

These examples demonstrate how to build a CliNER model which can then be used for predicting concepts in text files.

    cliner train --txt data/examples/ex_doc.txt --annotations data/examples/ex_doc.con --format i2b2 --model models/foo.model

This example trains a very simple CliNER model. The (pretend.txt, pretend.con) pair form as the only document for learning to identify concepts. We must specify that these files are i2b2 format (even though the .con extension implies i2b2 format, you can never be too careful). The CliNER model is then serialized to models/foo.model as specified.

**Please note that multiple files could be passed by enclosing them as a glob within "" quotes.**

(3) Prediction

Once your CliNER model is built, you can use it to predict concepts in text files.

    cliner predict --txt data/examples/ex_doc.txt --out data/test_predictions/ --format i2b2 --model models/foo.model

In this example, we use the models/foo.model CliNER model that we built up above. This model is used to predict concepts in i2b2 format for the "ex_doc.txt" file. This generates a file named "ex_doc.con" and stores it in the specified output directory.

(4) Evaluation

This allows us to evaluate how well CliNER does by comparing it against a gold standard.

    cliner evaluate --txt data/examples/ex_doc.txt --gold examples --predictions data/test_predictions/ --format i2b2

Evaluate how well the system predictions did. Both sets of data must be in the same format, and that format must be specified. This means that both the examples and data/test_predictions directories contain the file pretend.con.

Optional Resources
--------

There are a few external resources that are not packaged with CliNER but can improve prediction performance for feature extraction with the CRF.

**GENIA**

*Why would I want this?* 

The GENIA tagger is a tool similar to CliNER but designed for biomedical text. Depending on the domain of your data, this tool's pretrained model may or may not be able to improve performance for CliNER as it detects concepts.

**UMLS**

*Why would I want this?* 

The UMLS, or Unified Medical Language System, is a very comprehensive database of various medical terms and concepts. Access to it would allow CliNER to leverage domain-specific knowledge.

For installation of optional resources, please refer to the CliNER Wiki: 
    https://github.com/text-machine-lab/CliNER/wiki#optional-resources
