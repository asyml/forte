######################################################################
#  CliNER - documents.py                                             #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Build model for given training data.                     #
######################################################################

import os
import re

from examples.Cliner.CliNER.code.tools import clean_text, normalize_tokens

labels = {
    'O': 0,
    'B-problem': 1,
    'B-test': 2,
    'B-treatment': 3,
    'I-problem': 4,
    'I-test': 5,
    'I-treatment': 6,
}

id2tag = {v: k for k, v in labels.items()}


class Document:
    def __init__(self, txt, con=None):
        # read data
        retVal = read_i2b2(txt, con)

        # Internal representation natural for i2b2 format
        self._tok_sents = retVal[0]

        # Store token labels
        if con:
            self._tok_concepts = retVal[1]
            self._labels = tok_concepts_to_labels(self._tok_sents,
                                                  self._tok_concepts)

        # save filename
        self._filename = txt

    def getName(self):
        return os.path.basename(self._filename).split('.')[0]

    def getExtension(self):
        return 'con'

    def getTokenizedSentences(self):
        return self._tok_sents

    def getTokenLabels(self):
        return self._labels

    def conlist(self):
        return self._labels

    def write(self, pred_labels=None):
        """
        Purpose: Return the given concept label predictions in i2b2 format

        @param  pred_labels.     <list-of-lists> of predicted_labels
        @return                  <string> of i2b2-concept-file-formatted data
        """

        # Return value
        retStr = ''

        # If given labels to write, use them. Default to classifications
        if pred_labels is not None:
            token_labels = pred_labels
        elif self._labels is not None:
            token_labels = self._labels
        else:
            raise Exception('Cannot write concept file: must specify labels')

        concept_tuples = tok_labels_to_concepts(self._tok_sents, token_labels)

        # For each classification
        for classification in concept_tuples:

            # Ensure 'none' classifications are skipped
            if classification[0] == 'none':
                raise Exception(
                    'Classification label "none" should never happen')

            concept = classification[0]
            lineno = classification[1]
            start = classification[2]
            end = classification[3]

            # A list of words (corresponding line from the text file)
            text = self._tok_sents[lineno - 1]

            # print("\n" + "-" * 80)
            # print("classification: ", classification)
            # print("lineno:         ", lineno)
            # print("start:          ", start)
            # print("end             ", end)
            # print("text:           ", text)
            # print('len(text):      ', len(text))
            # print("text[start]:    ", text[start])
            # print("concept:        ", concept)

            datum = text[start]
            for j in range(start, end):
                datum += " " + text[j + 1]
            datum = datum.lower()

            # print('datum:          ', datum)

            # Line:TokenNumber of where the concept starts and ends
            idx1 = "%d:%d" % (lineno, start)
            idx2 = "%d:%d" % (lineno, end)

            # Classification
            label = concept

            # Print format
            retStr += "c=\"%s\" %s %s||t=\"%s\"\n" % (datum, idx1, idx2, label)

        # return formatted data
        return retStr.strip()


def read_i2b2(txt, con):
    """
    read_i2b2()

    @param txt. A file path for the tokenized medical record
    @param con. A file path for the i2b2 annotated concepts for txt
    """
    tokenized_sents = []

    sent_tokenize = lambda text: text.split('\n')
    word_tokenize = lambda text: text.split(' ')

    # Read in the medical text
    with open(txt) as f:
        # Original text file
        text = f.read().strip('\n')

        # tokenize
        sentences = sent_tokenize(text)
        for sentence in sentences:
            sent = clean_text(sentence.rstrip())

            # lowercase
            sent = sent.lower()

            toks = word_tokenize(sent)

            # normalize tokens
            normed_toks = normalize_tokens(toks)

            # for w in normed_toks:
            #    print(w)
            # print()

            tokenized_sents.append(normed_toks)

    # If an accompanying concept file was specified, read it
    tok_concepts = []
    if con:
        with open(con) as f:
            for line in f.readlines():
                # Empty line
                if not line.strip():
                    continue

                # parse concept line
                # pylint: disable=anomalous-backslash-in-string
                concept_regex = '^c="(.*)" (\\d+):(\\d+) (\\d+)' \
                                ':(\\d+)\\|\\|t="(.*)"$'
                match = re.search(concept_regex, line.strip())
                groups = match.groups()

                # retrieve regex info
                start_lineno = int(groups[1])
                start_tok_ind = int(groups[2])
                end_lineno = int(groups[3])
                end_tok_ind = int(groups[4])
                concept_label = groups[5]

                # pre-process text for error-check
                # matching_line = tokenized_sents[start_lineno-1]
                # matching_toks = matching_line[start_tok_ind:end_tok_ind+1]
                # matching_text = ' '.join(matching_toks).lower()
                # concept_text  = ' '.join(word_tokenize(concept_text))

                # error-check info
                assert start_lineno == end_lineno, 'concept must span single \
                line'

                # assert concept_text==matching_text, 'something wrong with \
                # inds'

                # add the concept info
                tup = (concept_label, start_lineno, start_tok_ind, end_tok_ind)
                tok_concepts.append(tup)

        # Safe guard against concept file having duplicate entries
        tok_concepts = list(set(tok_concepts))

        # Concept file does not guarantee ordering by line number
        tok_concepts = sorted(tok_concepts, key=lambda t: t[1:])

        # Ensure no overlapping concepts (that would be bad)
        for i in range(len(tok_concepts) - 1):
            c1 = tok_concepts[i]
            c2 = tok_concepts[i + 1]
            if c1[1] == c2[1]:
                if c1[2] <= c2[2] and c2[2] <= c1[3]:
                    fname = os.path.basename(con)
                    error1 = '%s has overlapping entities on line %d' % \
                             (fname, c1[1])
                    error2 = "It can't be processed until you remove one"
                    error3 = 'Please modify this file: %s' % con
                    error4 = '\tentity 1: c="%s" %d:%d %d:%d||t="%s"' % (
                        ' '.join(tokenized_sents[c1[1] - 1][c1[2]:c1[3] + 1]),
                        c1[1], c1[2], c1[1], c1[3], c1[0])
                    error5 = '\tentity 2: c="%s" %d:%d %d:%d||t="%s"' % (
                        ' '.join(tokenized_sents[c2[1] - 1][c2[2]:c2[3] + 1]),
                        c2[1], c2[2], c2[1], c2[3], c2[0])
                    error_msg = '\n\n%s\n%s\n\n%s\n\n%s\n%s\n' % (error1,
                                                                  error2,
                                                                  error3,
                                                                  error4,
                                                                  error5)
                    raise DocumentException(error_msg)

    return tokenized_sents, tok_concepts


def tok_concepts_to_labels(tokenized_sents, tok_concepts):
    # parallel to tokens
    # pylint: disable=redefined-outer-name
    labels = [['O' for tok in sent] for sent in tokenized_sents]

    # fill each concept's tokens appropriately
    for concept in tok_concepts:
        label, lineno, start_tok, end_tok = concept
        labels[lineno - 1][start_tok] = 'B-%s' % label
        for i in range(start_tok + 1, end_tok + 1):
            labels[lineno - 1][i] = 'I-%s' % label

    return labels


def tok_labels_to_concepts(tokenized_sents, tok_labels):
    '''
    for gold,sent in zip(tok_labels, tokenized_sents):
        print(gold)
        print(sent)
        print()
    '''

    # convert 'B-treatment' into ('B','treatment') and 'O' into ('O',None)
    def split_label(label):
        if label == 'O':
            iob, tag = 'O', None
        else:
            iob, tag = label.split('-')
        return iob, tag

    # preprocess predictions to "correct" starting Is into Bs
    corrected = []
    for lineno, lals in enumerate(tok_labels):
        corrected_line = []
        for _, i in enumerate(range(len(lals))):
            # '''
            # is this a candidate for error?
            iob, tag = split_label(lals[i])
            if iob == 'I':
                # beginning of line has no previous
                if i == 0:
                    print('CORRECTING! A')
                    new_label = 'B' + lals[i][1:]
                else:
                    # ensure either its outside OR mismatch type
                    prev_iob, prev_tag = split_label(lals[i - 1])
                    if prev_iob == 'O' or prev_tag != tag:
                        print('CORRECTING! B')
                        new_label = 'B' + lals[i][1:]
                    else:
                        new_label = lals[i]
            else:
                new_label = lals[i]
            # '''
            corrected_line.append(new_label)
        corrected.append(corrected_line)

    tok_labels = corrected

    concepts = []
    for i, labs in enumerate(tok_labels):

        N = len(labs)
        begins = [j for j, lab in enumerate(labs) if lab[0] == 'B']

        for start in begins:
            # "B-test"  -->  "-test"
            label = labs[start][1:]

            # get ending token index
            end = start
            while (end < N - 1) and tok_labels[i][end + 1].startswith('I') and \
                    tok_labels[i][end + 1][1:] == label:
                end += 1

            # concept tuple
            concept_tuple = (label[1:], i + 1, start, end)
            concepts.append(concept_tuple)

    # test it out
    test_tok_labels = tok_concepts_to_labels(tokenized_sents, concepts)
    # '''
    for lineno, (test, gold, sent) in enumerate(zip(test_tok_labels,
                                                    tok_labels,
                                                    tokenized_sents)):
        for i, (a, b) in enumerate(zip(test, gold)):
            # '''
            if not ((a == b) or (a[0] == 'B' and b[0] == 'I' and
                                 a[1:] == b[1:])):
                print()
                print('lineno:    ', lineno)
                print()
                print('generated: ', test[i - 3:i + 4])
                print('predicted: ', gold[i - 3:i + 4])
                print(sent[i - 3:i + 4])
                print('a[0]:  ', a[0])
                print('b[0]:  ', b[0])
                print('a[1:]: ', a[1:])
                print('b[1:]: ', b[1:])
                print('a[1:] == b[a:]: ', a[1:] == b[1:])
                print()
            # '''
            assert (a == b) or (a[0] == 'B' and b[0] == 'I' and a[1:] == b[1:])
            i += 1
    # '''
    assert test_tok_labels == tok_labels

    return concepts


class DocumentException(Exception):
    pass
