######################################################################
#  CliCon - genia_features.py                                        #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Independent GENIA module                                 #
######################################################################


from . import interface_genia
from .. import utils


class GeniaFeatures:

    def __init__(self, tagger, data):
        """
        Constructor.

        @param data. A list of split sentences
        """
        data = [[w for w in sent if w != ''] for sent in data]

        # Filter out nonprose sentences
        prose = [sent for sent in data if utils.is_prose_sentence(sent)]

        # Process prose sentences with GENIA tagger
        # self.GENIA_features = iter(interface_genia.genia(tagger, prose))
        self.gfeatures = {}
        gf = interface_genia.genia(tagger, prose)
        for sent, feats in zip(prose, gf):
            key = '%'.join(sent)
            self.gfeatures[key] = feats
        # self.GENIA_features = iter(interface_genia.genia(tagger, prose))

    def features(self, sentence):

        """
        features()

        @param sentence. A list of words to bind features to
        @return          list of dictionaries (of features)

        Note: All data is tagged upon instantiation of GeniaFeatures object.
              This function MUST take each line of the file (in order) as input
        """

        sentence = [w for w in sentence if w != '']

        # Mechanism to allow for skipping nonprose
        if not utils.is_prose_sentence(sentence):
            return []

        # Return value is a list of dictionaries (of features)
        features_list = [{} for _ in sentence]

        # print 'sentence: ', sentence
        # print 'len(sentence): ', len(sentence)

        # Get the GENIA features of the current sentence
        # genia_feats = next( self.GENIA_features )
        key = '%'.join(sentence)
        genia_feats = self.gfeatures[key]

        # print('\n\n\n')
        # print(len(sentence), len(genia_feats))
        for _, i in enumerate(range(len(sentence))):
            # print(i)
            # print(sentence[i])
            # print(genia_feats[i])
            # print()
            assert len(sentence[i]) == len(genia_feats[i]['GENIA-word'])
        # print 'genia_feats: ', [ f['GENIA-word'] for f in genia_feats ]
        # print 'len(genia_feats): ', len(genia_feats)
        assert len(sentence) == len(genia_feats)

        # Feature: Current word's GENIA features
        for i, curr in enumerate(genia_feats):
            assert curr['GENIA-word'] == sentence[i]
            keys = ['GENIA-stem', 'GENIA-POS', 'GENIA-chunktag']
            # keys = ['GENIA-stem','GENIA-POS','GENIA-chunktag', 'GENIA-NEtag']
            output = dict(((k, curr[k]), 1) for k in keys if k in curr)
            features_list[i].update(output)

        return features_list
