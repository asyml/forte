######################################################################
#  CliCon - umls_features.py                                         #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Independent UMLS module                                  #
######################################################################


from umls_cache import UmlsCache
import interpret_umls

umls_lookup_cache = UmlsCache()


def extract_umls_features(sentence):
    features_list = []
    for word in sentence:
        features_list.append(features_for_word(word))

    return features_list


def features_for_word(word):
    """
    UMLSFeatures::features_for_word()

    @ param word.  word to lookup in UMLS database
    @return        dictionary of  word-level features
    """

    # Return value is a list of dictionaries (of features)
    features = {}

    # Feature: UMLS Semantic Types
    cuis = interpret_umls.get_cui(umls_lookup_cache, word)
    if cuis:
        for cui in cuis:
            features[('umls_cui', cui)] = 1
    # Feature: UMLS Semantic Type (for each word)
    mapping = interpret_umls.umls_semantic_type_word(umls_lookup_cache, word)
    if mapping:
        for concept in mapping:
            features[('umls_semantc_type', concept)] = 1

    return features
