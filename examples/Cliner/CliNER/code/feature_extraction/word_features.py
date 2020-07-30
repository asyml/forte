######################################################################
#  CliNER - word_features.py                                         #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Isolate all word-level features into a single file       #
######################################################################


__author__ = 'Willie Boag'
__date__ = 'Apr 27, 2014'

import re

from nltk import LancasterStemmer, PorterStemmer

from .wordshape import getWordShapes

lancaster_st = LancasterStemmer()
porter_st = PorterStemmer()


def feature_word(word):
    return {('word', word.lower()): 1}


def feature_stem_lancaster(word):
    return {('stem_lancaster', lancaster_st.stem(word.lower())): 1}


def feature_generic(word):
    generic = re.sub('[0-9]', '0', word)
    return {('Generic#', generic): 1}


def feature_last_two_letters(word):
    return {('last_two_letters', word[-2:]): 1}


def feature_length(word):
    return {('length', ''): len(word)}


def feature_stem_porter(word):
    return {('stem_porter', porter_st.stem(word)): 1}


def feature_mitre(word):
    features = {}
    for f in mitre_features:
        if re.search(mitre_features[f], word):
            features[('mitre', f)] = 1
    return features


def feature_word_shape(word):
    features = {}
    wordShapes = getWordShapes(word)
    for shape in wordShapes:
        features[('word_shape', shape)] = 1
    return features


def feature_metric_unit(word):
    unit = ''
    if is_weight(word):
        unit = 'weight'
    elif is_size(word):
        unit = 'size'
    elif is_volume(word):
        unit = 'volume'
    return {('metric_unit', unit): 1}


def feature_prefix(word):
    prefix = word[:4].lower()
    return {("prefix", prefix): 1}


def QANN_features(word):
    """
    QANN_features()

    Purpose: Creates a dictionary of QANN features for the given word.

    @param word. A string
    @return      A dictionary of features

    >>> QANN_features('test') is not None
    True
    """

    features = {}

    # Feature: test result
    if is_test_result(word):
        features[('test_result', '')] = 1

    # Feature: measurements
    if is_measurement(word):
        features[('measurement', '')] = 1

    # Feature: directive
    if is_directive(word):
        features[('directive', '')] = 1

    # Feature: date
    if is_date(word):
        features[('date', '')] = 1

    # Feature: volume
    if is_volume(word):
        features[('volume', '')] = 1

    # Feature: weight
    if is_weight(word):
        features[('weight', '')] = 1

    # Feature: size
    if is_size(word):
        features[('size', '')] = 1

    # Feature: prognosis location
    # pylint: disable=using-constant-test
    if is_prognosis_location:
        features[('prog_location', '')] = 1

    # Feature: problem form
    if has_problem_form(word):
        features[('problem_form', '')] = 1

    # Feature: concept class
    if is_weight(word):
        features[('weight', '')] = 1

    return features


def feature_prev_word_stem(sentence, ind):
    if ind != 0:
        prev_ind = ind - 1
        prev_chunk = sentence[prev_ind].split()
        prev_word = porter_st.stem(prev_chunk[-1])
        return {('prev_word_stem', prev_word): 1}
    else:
        return {('prev_word_stem', '<START>'): 1}


def feature_next_word_stem(sentence, ind):
    if ind != len(sentence) - 1:
        next_ind = ind + 1
        next_chunk = sentence[next_ind].split()
        next_word = porter_st.stem(next_chunk[0])
        return {('next_word_stem', next_word): 1}
    else:
        return {('next_word_stem', '<END>'): 1}


enabled_IOB_prose_word_features = frozenset(
    [feature_generic, feature_last_two_letters, feature_word, feature_length,
     feature_stem_porter, feature_mitre,
     feature_stem_lancaster, feature_word_shape, feature_metric_unit])


def IOB_prose_features(word):
    """
    IOB_prose_features()

    Purpose: Creates a dictionary of prose  features for the given word.

    @param word. A string
    @return      A dictionary of features

    >>> IOB_prose_features('test') is not None
    True
    """

    # Feature: <dummy>
    features = {('dummy', ''): 1}  # always have >0 dimensions

    # Extract all enabled features
    for feature in enabled_IOB_prose_word_features:
        current_feat = feature(word)
        features.update(current_feat)

    return features


enabled_IOB_nonprose_word_features = frozenset(
    [feature_word, feature_word_shape, feature_mitre, QANN_features])


def IOB_nonprose_features(word):
    """
    IOB_nonprose_features()

    Purpose: Creates a dictionary of nonprose features for the given word.

    @param word. A string
    @return      A dictionary of features

    >>> IOB_nonprose_features('test') is not None
    True
    """

    # Feature: <dummy>
    features = {('dummy', ''): 1}  # always have >0 dimensions

    # Extract all enabled features
    for feature in enabled_IOB_nonprose_word_features:
        current_feat = feature(word)
        features.update(current_feat)

    return features


enabled_word_concept_features = frozenset(
    [feature_word, feature_prefix, feature_stem_porter, feature_stem_lancaster,
     feature_word_shape, feature_metric_unit,
     feature_mitre])


# Note: most of this function is currently commented out so the doctests
# should be fixed if this is ever changed
def concept_features_for_word(word):
    """
    concept_features_for_word()

    Purpose: Creates a dictionary of concept features for the given word.

    @param  word. A word to generate features for
    @return       A dictionary of features

    >>> concept_features_for_word('test') is not None
    True
    """

    features = {}

    # extract all selected features
    for feature in enabled_word_concept_features:
        current_feat = feature(word)
        features.update(current_feat)

    return features


enabled_chunk_concept_features = frozenset(
    [feature_prev_word_stem, feature_next_word_stem])


def concept_features_for_chunk(sentence, ind):
    """
    concept_features_for_chunk()

    @param  sentence    A sentence that has been chunked into vectors
            ind         The index of the concept in question within the
            sentence vector
    @return             A dictionary of features

    """

    features = {'dummy': 1}

    # Word-level features for each word of the chunk
    for w in sentence[ind].split():
        word_features = concept_features_for_word(w)
        features.update(word_features)

    # Context windows
    for feature in enabled_chunk_concept_features:
        current_feat = feature(sentence, ind)
        features.update(current_feat)

    return features


mitre_features = {
    "INITCAP": r"^[A-Z].*$",
    "ALLCAPS": r"^[A-Z]+$",
    "CAPSMIX": r"^[A-Za-z]+$",
    "HASDIGIT": r"^.*[0-9].*$",
    "SINGLEDIGIT": r"^[0-9]$",
    "DOUBLEDIGIT": r"^[0-9][0-9]$",
    "FOURDIGITS": r"^[0-9][0-9][0-9][0-9]$",
    "NATURALNUM": r"^[0-9]+$",
    "REALNUM": r"^[0-9]+.[0-9]+$",
    "ALPHANUM": r"^[0-9A-Za-z]+$",
    "HASDASH": r"^.*-.*$",
    "PUNCTUATION": r"^[^A-Za-z0-9]+$",
    "PHONE1": r"^[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]$",
    "PHONE2": r"^[0-9][0-9][0-9]-[0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]$",
    "FIVEDIGIT": r"^[0-9][0-9][0-9][0-9][0-9]",
    "NOVOWELS": r"^[^AaEeIiOoUu]+$",
    "HASDASHNUMALPHA": r"^.*[A-z].*-.*[0-9].*$ | *.[0-9].*-.*[0-9].*$",
    "DATESEPERATOR": r"^[-/]$",
}


# note: make spaces optional?
# Check about the documentation for this.
def is_test_result(context):
    """
    is_test_result()

    Purpose: Checks if the context is a test result.

    @param context. A string.
    @return         it returns the matching object of '[blank] was
    positive/negative' or None if it cannot find it.
                    otherwise, it will return True.

    >>> is_test_result('test was 10%')
    True
    >>> is_test_result('random string of words')
    None
    >>> is_test_result('Test')
    None
    >>> is_test_result('patient less than 30')
    True
    >>> is_test_result(' ')
    None
    """
    regex = r"^[A-Za-z]+( )*(-|--|:|was|of|\*|>|<|more than|less than)( )*[" \
            r"0-9]+(%)*"
    if not re.search(regex, context):
        return re.search(r"^[A-Za-z]+ was (positive|negative)", context)
    return True


def is_measurement(word):
    """
    is_measurement()

    Purpose: Checks if the word is a measurement.

    @param word. A string.
    @return      the matched object if it is a measurement, otherwise None.

    >>> is_measurement('10units') is not None
    True
    >>> is_measurement('7 units') is not None
    True
    >>> is_measurement('10cc') is not None
    True
    >>> is_measurement('300 L') is not None
    True
    >>> is_measurement('20mL') is not None
    True
    >>> is_measurement('400000 dL') is not None
    True
    >>> is_measurement('30000') is not None
    False
    >>> is_measurement('20dl') is not None
    False
    >>> is_measurement('units') is not None
    True
    """
    regex = r"^[0-9]*( )?(unit(s)|cc|L|mL|dL)$"
    return re.search(regex, word)


def is_directive(word):
    """
    is_directive()

    Purpose: Checks if the word is a directive.

    @param word. A string.
    @return      the matched object if it is a directive, otherwise None.

    >>> is_directive('q.abc') is not None
    True
    >>> is_directive('qAD') is not None
    True
    >>> is_directive('PRM') is not None
    True
    >>> is_directive('bid') is not None
    True
    >>> is_directive('prm') is not None
    True
    >>> is_directive('p.abc') is not None
    True
    >>> is_directive('qABCD') is not None
    False
    >>> is_directive('BID') is not None
    False
    """
    regex = r"^(q\..*|q..|PRM|bid|prm|p\..*)$"
    return re.search(regex, word)


def is_date(word):
    """
    is_date()

    Purpose: Checks if word is a date.

    @param word. A string.
    @return      the matched object if it is a date, otherwise None.

    >>> is_date('2015-03-1') is not None
    True
    >>> is_date('2014-02-19') is not None
    True
    >>> is_date('03-27-1995') is not None
    True
    >>> is_date('201') is not None
    False
    >>> is_date('0') is not None
    False
    """
    regex = r'^(\d\d\d\d-\d\d-\d|\d\d?-\d\d?-\d\d\d\d?|\d\d\d\d-\d\d?-\d\d?)$'
    return re.search(regex, word)


def is_volume(word):
    """
    is_volume()

    Purpose: Checks if word is a volume.

    @param word. A string.
    @return      the matched object if it is a volume, otherwise None.

    >>> is_volume('9ml') is not None
    True
    >>> is_volume('10 mL') is not None
    True
    >>> is_volume('552 dL') is not None
    True
    >>> is_volume('73') is not None
    False
    >>> is_volume('ml') is not None
    True
    """
    regex = r"^[0-9]*( )?(ml|mL|dL)$"
    return re.search(regex, word)


def is_weight(word):
    """
    is_weight()

    Purpose: Checks if word is a weight.

    @param word. A string.
    @return      the matched object if it is a weight, otherwise None.

    >>> is_weight('1mg') is not None
    True
    >>> is_weight('10 g') is not None
    True
    >>> is_weight('78 mcg') is not None
    True
    >>> is_weight('10000 milligrams') is not None
    True
    >>> is_weight('14 grams') is not None
    True
    >>> is_weight('-10 g') is not None
    False
    >>> is_weight('grams') is not None
    True
    """
    regex = r"^[0-9]*( )?(mg|g|mcg|milligrams|grams)$"
    return re.search(regex, word)


def is_size(word):
    """
    is_size()

    Purpose: Checks if the word is a size.

    @param word. A string.
    @return      the matched object if it is a weight, otheriwse None.

    >>> is_size('1mm') is not None
    True
    >>> is_size('10 cm') is not None
    True
    >>> is_size('36 millimeters') is not None
    True
    >>> is_size('423 centimeters') is not None
    True
    >>> is_size('328') is not None
    False
    >>> is_size('22 meters') is not None
    False
    >>> is_size('millimeters') is not None
    True
    """
    regex = r"^[0-9]*( )?(mm|cm|millimeters|centimeters)$"
    return re.search(regex, word)


def is_prognosis_location(word):
    """
    is_prognosis_location()

    Purpose: Checks if the word is a prognosis location

    @param word. A string.
    @return      the matched object if it is a prognosis location, otherwise
    None.

    >>> is_prognosis_location('c9-c5') is not None
    True
    >>> is_prognosis_location('C5-C9') is not None
    True
    >>> is_prognosis_location('test') is not None
    False
    >>> is_prognosis_location('c-9-C5') is not None
    False
    """
    regex = r"^(c|C)[0-9]+(-(c|C)[0-9]+)*$"
    return re.search(regex, word)


def has_problem_form(word):
    """
    has_problem_form()

    Purpose: Checks if the word has problem form.

    @param word. A string
    @return      the matched object if it has problem form, otheriwse None.

    >>> has_problem_form('prognosis') is not None
    True
    >>> has_problem_form('diagnosis') is not None
    True
    >>> has_problem_form('diagnostic') is not None
    True
    >>> has_problem_form('arachnophobic') is not None
    True
    >>> has_problem_form('test') is not None
    False
    >>> has_problem_form('ice') is not None
    False
    """
    regex = r".*(ic|is)$"
    return re.search(regex, word)


def get_def_class(word):
    """
    get_def_class()

    Purpose: Checks for a definitive classification at the word level.

    @param word. A string
    @return      1 if the word is a test term,
                 2 if the word is a problem term,
                 3 if the word is a treatment term,
                 0 otherwise.
    >>> get_def_class('eval')
    1
    >>> get_def_class('rate')
    1
    >>> get_def_class('tox')
    1
    >>> get_def_class('swelling')
    2
    >>> get_def_class('mass')
    2
    >>> get_def_class('broken')
    2
    >>> get_def_class('therapy')
    3
    >>> get_def_class('vaccine')
    3
    >>> get_def_class('treatment')
    3
    >>> get_def_class('unrelated')
    0
    """
    test_terms = {
        "eval", "evaluation", "evaluations",
        "sat", "sats", "saturation",
        "exam", "exams",
        "rate", "rates",
        "test", "tests",
        "xray", "xrays",
        "screen", "screens",
        "level", "levels",
        "tox"
    }
    problem_terms = {
        "swelling",
        "wound", "wounds",
        "symptom", "symptoms",
        "shifts", "failure",
        "insufficiency", "insufficiencies",
        "mass", "masses",
        "aneurysm", "aneurysms",
        "ulcer", "ulcers",
        "trama", "cancer",
        "disease", "diseased",
        "bacterial", "viral",
        "syndrome", "syndromes",
        "pain", "pains"
                "burns", "burned",
        "broken", "fractured"
    }
    treatment_terms = {
        "therapy",
        "replacement",
        "anesthesia",
        "supplement", "supplemental",
        "vaccine", "vaccines"
                   "dose", "doses",
        "shot", "shots",
        "medication", "medicine",
        "treatment", "treatments"
    }
    if word.lower() in test_terms:
        return 1
    elif word.lower() in problem_terms:
        return 2
    elif word.lower() in treatment_terms:
        return 3
    return 0
