######################################################################
#  CliCon - clicon_genia_interface.py                                #
#                                                                    #
#  Willie Boag                                      wboag@cs.uml.edu #
#                                                                    #
#  Purpose: Provide a way for Python to utilize the output of the    #
#               GENIA Tagger                                         #
#                                                                    #
#  Genia Tagger: http://www.nactem.ac.uk/tsujii/GENIA/tagger/        #
######################################################################


__author__ = 'Willie Boag'
__date__ = 'Jan. 27, 2014'

import os
import tempfile
# from commands import getstatusoutput
from subprocess import Popen, PIPE

from .genia_cache import GeniaCache

cliner_dir = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
tmp_dir = os.path.join(cliner_dir, 'data', 'tmp')


def genia(geniatagger, data):
    '''
    genia()

    Purpose: Call the genia tagger and return its output in python format

    @param geniatagger.  A path to the executable geniatagger
    @param data.         A list of list of strings (lines of words from a file)
    @return              A list of dcitionaries of the genia tagger's output.
    '''

    # Lookup cache
    cache = GeniaCache()

    # Get uncached lines
    uncached = []
    for line in data:
        sent = ' '.join(line)
        if not cache.has_key(sent):
            uncached.append(sent)

    if uncached:
        # write list to file and then feed it to GENIA
        genia_dir = os.path.dirname(geniatagger)

        os_handle, out = tempfile.mkstemp(dir=tmp_dir, suffix="genia_temp")

        with open(out, 'w') as f:
            for line in uncached:
                f.write(line + '\n')

        # Run genia tagger
        print('\t\tRunning  GENIA tagger')
        genia_dir = os.path.dirname(geniatagger)
        # stream = getstatusoutput('cd %s ; ./geniatagger -nt %s' %(
        # genia_dir,out))
        p = Popen('cd %s ; ./geniatagger -nt %s' % (genia_dir, out), shell=True,
            stdout=PIPE, stderr=PIPE)
        stream_b, _ = p.communicate()

        stream = stream_b.decode('ascii')

        # print '\t\tFinished GENIA tagger'

        # Organize tagger output
        linetags = []
        tagged = []

        # if the sentence is too long genia outputs an error.
        stream_lines = stream.split('\n')

        # print('\n\n\n')
        # print(stream_lines)
        # print('\n\n\n')

        # get the line the warning might be on.
        # potential_warning = "" if len(stream_lines[4:5]) == 0 else
        # stream_lines[4:5][0]

        genia_stream = None

        # genia_stream = stream_lines[4:]
        genia_stream = stream_lines

        for tag in genia_stream:
            if tag.startswith('warning: the sentence seems to be too long'):
                print('WARNING:', tag)
                continue

            if tag.split():  # Part of line
                linetags.append(tag)
            else:  # End  of line
                tagged.append(linetags)
                linetags = []

        # Add tagger output to cache
        for line, tags in zip(uncached, tagged):
            # print(line)
            for w, feat in zip(line.split(), tags):
                # print('\t', w, feat.split('\t')[0])
                assert w == feat.split('\t')[0]
            # print('\n\n\n')
            cache.add_map(line, tags)
        # print('-'*80)

        # Remove temp file
        os.close(os_handle)

        # print 'GENIA OUTPUT: ', open(out,"rb").read()

        os.remove(out)

    for sent in data:
        feats = cache.get_map(' '.join(sent))
        # print(sent)
        for w, feat in zip(sent, feats):
            # print('\t', w, feat.split('\t')[0])
            assert w == feat.split('\t')[0]
        # print('\n\n\n')
    # exit()

    # Extract features
    linefeats = []
    retlist = []
    for line in data:

        # print 'line: ', line

        line = ' '.join(line)

        # Get tagged output from cache
        tags = cache.get_map(line)

        # print 'tags: ', tags

        for tag in tags:
            tag = tag.split()
            output = {'GENIA-word': tag[0],
                      'GENIA-stem': tag[1],
                      'GENIA-POS': tag[2],
                      'GENIA-chunktag': tag[3],
                      'GENIA-NEtag': tag[4]}

            linefeats.append(output)

        retlist.append(linefeats)
        linefeats = []

    # print 'retlist: ', retlist

    return retlist
