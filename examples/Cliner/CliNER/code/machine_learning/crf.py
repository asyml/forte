######################################################################
#  CliCon - crf.py                                                   #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Implement CRF (using python-crfsuite)                    #
######################################################################

import os
import tempfile

import pycrfsuite

from examples.Cliner.CliNER.code.feature_extraction.read_config \
    import enabled_modules
from examples.Cliner.CliNER.code.tools import compute_performance_stats

cliner_dir = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
tmp_dir = os.path.join(cliner_dir, 'data', 'tmp')


def format_features(rows, labels=None):
    retVal = []

    # For each line
    for i, line in enumerate(rows):

        # For each word in the line
        for j, features in enumerate(line):

            # Nonzero dimensions
            inds = features.nonzero()[1]

            # If label exists
            values = []
            if labels:
                values.append(str(labels[i][j]))

            # Value for each dimension
            for k in inds:
                values.append('%d=%d' % (k, features[0, k]))

            retVal.append("\t".join(values).strip())

        # Sentence boundary seperator
        retVal.append('')

    return retVal


def pycrf_instances(fi, labeled):
    xseq = []
    yseq = []

    # Skip first element
    if labeled:
        begin = 1
    else:
        begin = 0

    for line in fi:
        line = line.strip('\n')
        if not line:
            # An empty line presents an end of a sequence.
            if labeled:
                yield xseq, tuple(yseq)
            else:
                yield xseq

            xseq = []
            yseq = []
            continue

        # Split the line with TAB characters.
        fields = line.split('\t')

        # Append the item to the item sequence.
        feats = fields[begin:]
        xseq.append(feats)

        # Append the label to the label sequence.
        if labeled:
            yseq.append(fields[0])


def train(X, Y, val_X=None, val_Y=None, test_X=None, test_Y=None):

    # Sanity Check detection: features & label
    # with open('a','w') as f:
    #    for xline,yline in zip(X,Y):
    #        for x,y in zip(xline,yline):
    #            print >>f, y, '\t', x.nonzero()[1][0]
    #        print >>f

    # Format features fot crfsuite
    feats = format_features(X, Y)

    # Create a Trainer object.
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in pycrf_instances(feats, labeled=True):
        trainer.append(xseq, yseq)

    # Train the model
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    os_handle, tmp_file = tempfile.mkstemp(dir=tmp_dir, suffix="crf_temp")
    trainer.train(tmp_file)

    # Read the trained model into a string (so it can be pickled)
    model = ''
    with open(tmp_file, 'rb') as f:
        model = f.read()
    os.close(os_handle)

    # Remove the temporary file
    os.remove(tmp_file)

    ######################################################################

    # information about fitting the model
    scores = {}

    # how well does the model fir the training data?
    train_pred = predict(model, X)
    train_stats = compute_performance_stats('train', train_pred, Y)
    scores['train'] = train_stats

    if val_X:
        val_pred = predict(model, val_X)
        val_stats = compute_performance_stats('dev', val_pred, val_Y)
        scores['dev'] = val_stats

    if test_X:
        test_pred = predict(model, test_X)
        test_stats = compute_performance_stats('test', test_pred, test_Y)
        scores['test'] = test_stats

    # keep track of which external modules were used for building this model!
    scores['hyperparams'] = {}
    enabled_mods = enabled_modules()
    for module, enabled in enabled_mods.items():
        e = bool(enabled)
        scores['hyperparams'][module] = e

    return model, scores


def predict(clf, X):
    # Format features fot crfsuite
    feats = format_features(X)

    # Dump the model into a temp file
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    os_handle, tmp_file = tempfile.mkstemp(dir=tmp_dir, suffix="crf_temp")
    with open(tmp_file, 'wb') as f:
        clf_byte = bytearray(clf)  # , 'latin1')
        f.write(clf_byte)

    # Create the Tagger object
    tagger = pycrfsuite.Tagger()
    tagger.open(tmp_file)

    # Remove the temp file
    os.close(os_handle)
    os.remove(tmp_file)

    # Tag the sequence
    retVal = []
    Y = []
    for xseq in pycrf_instances(feats, labeled=False):
        yseq = [int(n) for n in tagger.tag(xseq)]
        retVal += list(yseq)
        Y.append(list(yseq))
    # Sanity Check detection: feature & label predictions
    # with open('a','w') as f:
    #    for x,y in zip(xseq,Y):
    #        x = x[0]
    #        print >>f, y, '\t', x[:-2]
    #    print >>f

    return Y
