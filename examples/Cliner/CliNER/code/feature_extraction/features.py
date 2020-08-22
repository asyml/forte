######################################################################
#  CliCon - features.py                                              #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Isolate the model's features from model.py               #
######################################################################


from examples.Cliner.CliNER.code.feature_extraction import \
    word_features as feat_word
from examples.Cliner.CliNER.code.feature_extraction.read_config import \
    enabled_modules
# What modules are available
from examples.Cliner.CliNER.code.feature_extraction.utils import load_pos_tagger

################################################
# Build a few expensive one-time objects


# what to build requires knowing what tools are enabled
enabled = enabled_modules()

# Import feature modules
feat_genia = None
if enabled['GENIA']:
    from examples.Cliner.CliNER.code.feature_extraction.genia_dir\
        .genia_features import \
        GeniaFeatures

# Only create UMLS cache if module is available
if enabled['UMLS']:
    import umls_dir.umls_features as feat_umls
    from examples.Cliner.CliNER.code.feature_extraction.umls_dir.umls_cache \
        import UmlsCache

    umls_cache = UmlsCache()

# POS tagger
nltk_tagger = load_pos_tagger()

################################################


# which features are enabled
enabled_IOB_prose_sentence_features = []
enabled_IOB_prose_sentence_features.append('unigram_context')
enabled_IOB_prose_sentence_features.append('pos')
enabled_IOB_prose_sentence_features.append('pos_context')
enabled_IOB_prose_sentence_features.append('prev')
enabled_IOB_prose_sentence_features.append('prev2')
enabled_IOB_prose_sentence_features.append('next')
enabled_IOB_prose_sentence_features.append('next2')
enabled_IOB_prose_sentence_features.append('GENIA')
enabled_IOB_prose_sentence_features.append('UMLS')


def extract_features(tok_sents):
    """
    extract_features()
    @param data      A list of split sentences (1 sent = 1 line from file)
    @param Y         A list of list of IOB (1:1 mapping with data)
    @return          tuple: list of IOB_prose_features, list of IOB
    """
    # Genia preprocessing
    sentence_features_preprocess(tok_sents)

    # iterate through all data & extract features (sentence-by-sentence)
    prose_feats = []
    for sentence in tok_sents:
        prose_feats.append(extract_features_sentence(sentence))
    return prose_feats


def sentence_features_preprocess(data):
    # pylint: disable=global-statement
    global feat_genia
    tagger = enabled['GENIA']
    # Only run GENIA tagger if module is available
    if tagger:
        feat_genia = GeniaFeatures(tagger, data)


def extract_features_sentence(sentence):
    """
    extract_features_sentence
    Compute a list of dict-based feature representation for a list of tokens.
    @param sentence. A list of tokens.
    @return          A list of feature dictionaries.
    """
    features_list = []

    # Get a feature set for each word in the sentence
    for i, _ in enumerate(sentence):
        features_list.append(feat_word.IOB_prose_features(sentence[i]))

    # Feature: Bag of Words unigram conext (window=3)
    if 'unigram_context' in enabled_IOB_prose_sentence_features:
        window = 3
        n = len(sentence)

        # Previous unigrams
        for i in range(n):
            end = min(i, window)
            unigrams = sentence[i - end:i]
            for j, u in enumerate(unigrams):
                features_list[i][('prev_unigrams-%d' % j, u)] = 1

        # Next     unigrams
        for i in range(n):
            end = min(i + window, n - 1)
            unigrams = sentence[i + 1:end + 1]
            for j, u in enumerate(unigrams):
                features_list[i][('next_unigrams-%d' % j, u)] = 1

    # Only POS tag once
    if 'pos' in enabled_IOB_prose_sentence_features:
        pos_tagged = nltk_tagger.tag(sentence)

    # Allow for particular features to be enabled
    for feature in enabled_IOB_prose_sentence_features:

        # Feature: Part of Speech
        if feature == 'pos':
            for (i, (_, pos)) in enumerate(pos_tagged):
                features_list[i].update({('pos', pos): 1})

        # Feature: POS context
        if 'pos_context' in enabled_IOB_prose_sentence_features:
            window = 3
            n = len(sentence)

            # Previous POS
            for i in range(n):
                end = min(i, window)
                for j, p in enumerate(pos_tagged[i - end:i]):
                    pos = p[1]
                    features_list[i][('prev_pos_context-%d' % j, pos)] = 1

            # Next POS
            for i in range(n):
                end = min(i + window, n - 1)
                for j, p in enumerate(pos_tagged[i + 1:i + end + 1]):
                    pos = p[1]
                    features_list[i][('prev_pos_context-%d' % j, pos)] = 1

        # GENIA features
        if (feature == 'GENIA') and enabled['GENIA']:

            # Get GENIA features
            genia_feat_list = feat_genia.features(sentence)

            for i, feat_dict in enumerate(genia_feat_list):
                features_list[i].update(feat_dict)

        # Feature: UMLS Word Features (only use prose ones)
        if (feature == "UMLS") and enabled['UMLS']:
            umls_features = feat_umls.extract_umls_features(sentence)
            for i in range(len(sentence)):
                features_list[i].update(umls_features[i])

    #######
    # TODO: This section is ugly... factorize it.
    #######

    # Used for 'prev' and 'next' features
    ngram_features = [{} for i in range(len(features_list))]
    if "prev" in enabled_IOB_prose_sentence_features:
        prev = lambda f: {("prev_" + k[0], k[1]): v for k, v in f.items()}
        prev_list = list(map(prev, features_list))
        for i in range(len(features_list)):
            if i == 0:
                ngram_features[i][("prev", "*")] = 1
            else:
                ngram_features[i].update(prev_list[i - 1])

    if "prev2" in enabled_IOB_prose_sentence_features:
        prev2 = lambda f: {("prev2_" + k[0], k[1]): v / 2.0 for k, v in
                           f.items()}
        prev_list = list(map(prev2, features_list))
        for i in range(len(features_list)):
            if i == 0:
                ngram_features[i][("prev2", "*")] = 1
            elif i == 1:
                ngram_features[i][("prev2", "*")] = 1
            else:
                ngram_features[i].update(prev_list[i - 2])

    if "next" in enabled_IOB_prose_sentence_features:
        next = lambda f: {("next_" + k[0], k[1]): v for k, v in f.items()}
        next_list = list(map(next, features_list))
        for i in range(len(features_list)):
            if i < len(features_list) - 1:
                ngram_features[i].update(next_list[i + 1])
            else:
                ngram_features[i][("next", "*")] = 1

    if "next2" in enabled_IOB_prose_sentence_features:
        next2 = lambda f: {("next2_" + k[0], k[1]): v / 2.0 for k, v in
                           f.items()}
        next_list = list(map(next2, features_list))
        for i in range(len(features_list)):
            if i < len(features_list) - 2:
                ngram_features[i].update(next_list[i + 2])
            elif i == len(features_list) - 2:
                ngram_features[i][("next2", "**")] = 1
            else:
                ngram_features[i][("next2", "*")] = 1

    merged = lambda d1, d2: dict(list(d1.items()) + list(d2.items()))
    features_list = [merged(features_list[i], ngram_features[i])
                     for i in range(len(features_list))]

    return features_list


def display_enabled_modules():
    print()
    for module, status in enabled.items():
        if status:
            print('\t', module, '\t', ' ENABLED')
        else:
            print('\t', module, '\t', 'DISABLED')
    print()
