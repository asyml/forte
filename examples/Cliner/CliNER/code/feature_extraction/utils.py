######################################################################
#  CliCon - utilities.py                                             #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Miscellaneous tools for handling data.                   #
######################################################################


import os
import pickle
import re
import sys

import nltk

# used as a default path for stashing pos tagger.
dname = os.path.dirname
CLINER_DIR = dname(dname(dname(os.path.abspath(__file__))))
tagger_name = 'py%d_maxent_treebank_pos_tagger.pickle' % sys.version_info.major
pos_tagger_path = os.path.join(CLINER_DIR, 'tools', tagger_name)


def load_pickled_obj(path_to_pickled_obj):
    data = None
    with open(path_to_pickled_obj, "rb") as f:
        data = f.read()
    return pickle.loads(data)


def pickle_dump(obj, path_to_obj):
    f = open(path_to_obj, "wb")
    # NOTE: using highest priority makes loading TRAINED models load really
    # slowly.
    # use this function for anything BUT THAT!. I mainly made this for
    # loading pos tagger...
    pickle.dump(obj, f, -1)
    f.close()


def dump_pos_tagger(path_to_obj):
    # pylint: disable=protected-access
    tagger = nltk.data.load(nltk.tag._POS_TAGGER)
    pickle_dump(tagger, path_to_obj)


def load_pos_tagger(path_to_obj=pos_tagger_path):
    tagger = load_pickled_obj(path_to_obj)
    return tagger


def is_prose_sentence(sentence):
    """
    is_prose_sentence()

    Purpose: Determine if a sentence of text is 'prose'

    @param sentence A list of words
    @return         A boolean

    >>> is_prose_sentence(['Admission', 'Date', ':'])
    False
    >>> is_prose_sentence(['Hello', 'World', '.'])
    True
    >>> is_prose_sentence(['What', 'do', 'you', 'think', '?'])
    True
    >>> is_prose_sentence(['Short', 'sentence'])
    False
    """
    # Empty sentence is not prose
    if not sentence:
        return False

    if sentence[-1] == '.' or sentence[-1] == '?':
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
    """
    is_at_least_half_nonprose(sentence)

    Purpose: Checks if at least half of the sentence is considered to be
    'nonprose'

    @param sentence. A list of words
    @return          A boolean

    >>> is_at_least_half_nonprose(['1','2','and','some','words'])
    True
    >>> is_at_least_half_nonprose(['1', '2', '3', '4', 'and', 'some',
    'words', '5'])
    False
    >>> is_at_least_half_nonprose(['word'])
    True
    >>> is_at_least_half_nonprose([' '])
    True
    """
    count = len([w for w in sentence if is_prose_word(w)])

    return bool(count >= len(sentence) / 2)
    #     return True
    # else:
    #     return False


def is_prose_word(word):
    """
    is_prose_word(word)

    Purpose: Checks if the given word is 'prose'

    @param word. A word
    @return      A boolean

    >>> is_prose_word('word')
    True
    >>> is_prose_word('99')
    False
    >>> is_prose_word('question?')
    False
    >>> is_prose_word('ALLCAPS')
    False
    """
    # Punctuation
    for punc in ".?,!:\"'":
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

# EOF
