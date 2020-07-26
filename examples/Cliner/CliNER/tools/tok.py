import sys
import re
import nltk


def main():
    file_tokenize(sys.argv[1], sys.argv[2])


def file_tokenize(filename, outfile):
    toks = tokenize(filename)
    with open(outfile, 'w') as f:
        for sent in toks:
            print >>f, ' '.join(sent)


def tokenize(filename):
    with open(filename, 'r') as f:
        text = f.read().strip()
    text = clean_text(text)

    text = re.sub('\n\n+', '\n\n', text)

    # remove PHI
    phis = re.findall('(\[\*\*.*?\*\*\])', text)
    for phi in phis:
        new = replace_phi(text, phi)
        text = text.replace(phi, new)

    # break into sentences
    sections = text.split('\n\n')
    sents = []
    for section in sections:
        # remove leading section lines
        if '\n' in section:
            index = section.index('\n')
        else:
            index = 500
        while index < 60:
            # add that line to all sents
            line = section[:index]
            sents.append(line)

            section = section[index+1:]
            if '\n' in section:
                index = section.index('\n')
            else:
                index = 500

        s_toks = nltk.sent_tokenize(section)
        sents += s_toks

    # break into words
    word_toks = []
    for sent in sents:
        w_toks = nltk.word_tokenize(sent)
        word_toks.append(w_toks)

    return word_toks


def clean_text(text):
    try:
        return text.decode('ascii', 'ignore')
    except UnicodeDecodeError, e:
        chars = []
        for c in text:
            try:
                c.decode('ascii', 'ignore')
                chars.append(c)
            except UnicodeDecodeError, f:
                pass
        return ''.join(chars)


def replace_phi(text, phi):
    return '__phi__'


if __name__ == '__main__':
    main()

