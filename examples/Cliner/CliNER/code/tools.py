######################################################################
#  CliNER - tools.py                                                 #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: General purpose tools                                    #
######################################################################


import errno
import math
import os
import pickle
import re
import string
import sys

import numpy as np


#############################################################
#  files
#############################################################

def map_files(files):
    """Maps a list of files to basename -> path."""
    output = {}
    for f in files:  # pylint: disable=invalid-name
        basename = os.path.splitext(os.path.basename(f))[0]
        output[basename] = f
    return output


def mkpath(path):
    """Alias for mkdir -p."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


#############################################################
#  text pre-processing
#############################################################

def clean_text(text):
    return ''.join(map(lambda x: x if (x in string.printable) else '@', text))


def normalize_tokens(toks):
    # todo: normalize dosages (icluding 8mg -> mg)
    # replace number tokens
    def num_normalize(w):
        # pylint: disable=anomalous-backslash-in-string
        return '__num__' if re.search('\\d', w) else w

    toks = list(map(num_normalize, toks))
    return toks


#############################################################
#  manipulating list-of-lists
#############################################################

def flatten(list_of_lists):
    '''
    flatten()
    Purpose: Given a list of lists, flatten one level deep
    @param list_of_lists. <list-of-lists> of objects.
    @return               <list> of objects (AKA flattened one level)
    >>> flatten([['a','b','c'],['d','e'],['f','g','h']])
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    '''
    return sum(list_of_lists, [])


def save_list_structure(list_of_lists):
    '''
    save_list_structure()
    Purpose: Given a list of lists, save way to recover structure from
    flattended
    @param list_of_lists. <list-of-lists> of objects.
    @return               <list> of indices, where each index refers to the
                                 beginning of a line in the orig list-of-lists
    >>> save_list_structure([['a','b','c'],['d','e'],['f','g','h']])
    [3, 5, 8]
    '''

    offsets = [len(sublist) for sublist in list_of_lists]
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i - 1]

    return offsets


def reconstruct_list(flat_list, offsets):
    '''
    save_list_structure()
    Purpose: This undoes a list flattening. Uses value from
    save_list_structure()
    @param flat_list. <list> of objects
    @param offsets    <list> of indices, where each index refers to the
                                 beginning of a line in the orig list-of-lists
    @return           <list-of-lists> of objects (the original structure)
    >>> reconstruct_list(['a','b','c','d','e','f','g','h'], [3,5,8])
    [['a', 'b', 'c'], ['d', 'e'], ['f', 'g', 'h']]
    '''

    return [flat_list[i:j] for i, j in zip([0] + offsets, offsets)]


#############################################################
#  serialization to disc
#############################################################

def load_pickled_obj(path_to_pickled_obj):
    data = None
    with open(path_to_pickled_obj, "rb") as f:
        data = f.read()
    return pickle.loads(data)


def pickle_dump(obj, path_to_obj):
    # NOTE: highest priority makes loading TRAINED models slow
    with open(path_to_obj, 'wb') as f:
        pickle.dump(obj, f, -1)


#############################################################
#  prose v nonprose
#############################################################


def is_prose_sentence(sentence):
    # pylint: disable=unidiomatic-typecheck
    assert isinstance(sentence) == isinstance([]), 'is_prose_sentence() must take list arg'
    if sentence == []:
        return False
    # elif sentence[-1] == '.' or sentence[-1] == '?':
    elif sentence[-1] == '?':
        return True
    elif sentence[-1] == ':':
        return False
    elif len(sentence) <= 5:
        return False
    elif is_at_least_half_nonprose(sentence):
        return True
    else:
        return False


def is_at_least_half_nonprose(sentence):
    count = len(filter(is_prose_word, sentence))
    return bool(count >= len(sentence) / 2)


def is_prose_word(word):
    # Punctuation
    for punc in string.punctuation:
        if punc in word:
            return False
    # Digit
    # pylint: disable=anomalous-backslash-in-string
    if re.match('\\d', word):
        return False
    # All uppercase
    if word == word.upper():
        return False
    # Else
    return True


def prose_partition(tokenized_sents, labels=None):
    prose_sents = []
    nonprose_sents = []
    prose_labels = []
    nonprose_labels = []

    # partition the sents & labels into EITHER prose OR nonprose groups
    for _, i in enumerate(range(len(tokenized_sents))):
        if is_prose_sentence(tokenized_sents[i]):
            prose_sents.append(tokenized_sents[i])
            if labels:
                prose_labels.append(labels[i])
        else:
            nonprose_sents.append(tokenized_sents[i])
            if labels:
                nonprose_labels.append(labels[i])

    # group data appropriately (note, labels might not be provided)
    if labels:
        prose = (prose_sents, prose_labels)
        nonprose = (nonprose_sents, nonprose_labels)
    else:
        prose = (prose_sents, None)
        nonprose = (nonprose_sents, None)

    return prose, nonprose


def print_files(f, file_names):
    COLUMNS = 4
    file_names = sorted(file_names)
    start = 0
    for _ in range(int(math.ceil(float(len(file_names)) / COLUMNS))):
        write(f, u'\t\t')
        for featname in file_names[start:start + COLUMNS]:
            write(f, '%-15s' % featname)
        write(f, u'\n')
        start += COLUMNS


# python2 needs to convert to unicdode, but thats default for python3
if sys.version_info.major == 3:
    tostr = str


def write(f, s):
    f.write(tostr(s))


def print_vec(f, label, vec):
    '''
    print_vec()

    Pretty formatting for displaying a vector of numbers in a log.

    @param f.           An open file stream to write to.
    @param label.  A description of the numbers (e.g. "recall").
    @param vec.    A numpy array of the numbers to display.
    '''
    COLUMNS = 7
    start = 0
    write(f, '\t%-10s: ' % label)
    # pylint: disable=unidiomatic-typecheck
    if isinstance(vec) != isinstance([]):
        vec = vec.tolist()
    for _ in range(int(math.ceil(float(len(vec)) / COLUMNS))):
        for featname in vec[start:start + COLUMNS]:
            write(f, '%7.3f' % featname)
        write(f, u'\n')
        start += COLUMNS


def print_str(f, label, names):
    '''
    print_str()
    Pretty formatting for displaying a list of strings in a log
    @param f.           An open file stream to write to.
    @param label.  A description of the numbers (e.g. "recall").
    @param names.  A list of strings.
    '''
    COLUMNS = 4
    start = 0
    for row in range(int(math.ceil(float(len(names)) / COLUMNS))):
        if row == 0:
            write(f, '\t%-10s: ' % label)
        else:
            write(f, '\t%-10s  ' % '')

        for featname in names[start:start + COLUMNS]:
            write(f, '%-16s ' % featname)

        write(f, u'\n')
        start += COLUMNS


#############################################################
#  Quick-and-Dirty evaluation of performance
#############################################################


def compute_performance_stats(label, pred, ref):
    '''
    compute_stats()
    Compute the P, R, and F for a given model on some data.
    @param label.  A name for the data (e.g. "train" or "dev")
    @param pred.   A list of list of predicted labels.
    @param pred.   A list of list of true      labels.
    '''

    num_tags = max(set(sum(ref, [])) | set(sum(pred, []))) + 1
    # confusion matrix
    confusion = np.zeros((num_tags, num_tags))
    for tags, yseq in zip(pred, ref):
        for y, p in zip(yseq, tags):
            confusion[p, y] += 1

    # print confusion matrix
    conf_str = ''
    conf_str += '\n\n'
    conf_str += label + '\n'
    conf_str += ' ' * 6
    for i in range(num_tags):
        conf_str += '%4d ' % i
    conf_str += ' (gold)\n'
    for i in range(num_tags):
        conf_str += '%2d    ' % i
        for j in range(num_tags):
            conf_str += '%4d ' % confusion[i][j]
        conf_str += '\n'
    conf_str += '(pred)\n'
    conf_str += '\n\n'
    # print conf_str

    precision = np.zeros(num_tags)
    recall = np.zeros(num_tags)
    f1 = np.zeros(num_tags)

    for i in range(num_tags):
        correct = confusion[i, i]
        num_pred = sum(confusion[i, :])
        num_actual = sum(confusion[:, i])

        p = correct / (num_pred + 1e-9)
        r = correct / (num_actual + 1e-9)

        precision[i] = p
        recall[i] = r
        f1[i] = (2 * p * r) / (p + r + 1e-9)

    scores = {}
    scores['precision'] = precision
    scores['recall'] = recall
    scores['f1'] = f1
    scores['conf'] = conf_str

    return scores
