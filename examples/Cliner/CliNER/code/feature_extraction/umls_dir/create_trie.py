
#database.py creates a .db file for performing umls searches.
import marisa_trie
import sys
import os
import atexit

features_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if features_dir not in sys.path:
    sys.path.append(features_dir)

# find where umls tables are located
from read_config import enabled_modules
enabled = enabled_modules()
umls_tables = enabled['UMLS']

trie_path = None
success = False
MRCON_TABLE = None

@atexit.register
def trie_cleanup():

    global trie_path
    global MRCON_TABLE
    global success

    if success is False:

        print >>sys.stderr, '\n\tError: trie was not created succesfully.\n'

        if trie_path is not None:

            try:
                os.remove(trie_path)
            except:
                pass

    if MRCON_TABLE is not None:
        MRCON_TABLE.close()


def create_trie():

    global trie_path
    global MRCON_TABLE
    global success

    """
    create_trie()

    Purpose: Build a trie of concepts from MRREL

    @return  A trie object
    """
    # Is trie already built & pickled?
    trie_path = os.path.join(umls_tables, 'umls-concept.trie')
    try:
        t = marisa_trie.Trie().load(trie_path)
        success = True
        return t
    except IOError:
        pass


    print "\ncreating concept-trie"

    #load data in files.
    print "opening file"
    try:
        mrcon_path = os.path.join(umls_tables, 'MRCONSO.RRF')
        MRCON_TABLE = open( mrcon_path , "r" )
    except IOError:
        print "\nNo file to use for creating MRCON table\n"
        sys.exit()

    print "inserting data into concept-trie"

    #insert data onto database
    print "inserting data"
    concepts = []
    for line in MRCON_TABLE:

        line = line.split('|')
        line.pop()

        assert len(line) == 18

        if len(line) < 6: continue

        concept = line[14]

        # Ignore non-ascii
        try:
            concept.decode('ascii')
        except:
            continue

        #print type(concept)
        concepts.append(concept)

    print "creating trie"
    t = marisa_trie.Trie(concepts)

    print "concept-trie created"

    # Pickle trie

    t.save(trie_path)

    success = True

    return t


if __name__ == '__main__':
    t = create_trie()
