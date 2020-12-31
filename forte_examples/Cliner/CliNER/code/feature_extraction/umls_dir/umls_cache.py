import atexit
import os
import pickle
import sys

from read_config import enabled_modules
from utilities import load_pickled_obj

features_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if features_dir not in sys.path:
    sys.path.append(features_dir)

# find where umls tables are located

enabled = enabled_modules()
umls_tables = enabled['UMLS']


class UmlsCache:
    # static class variables
    filename = None
    cache = None

    def __init__(self):

        try:

            UmlsCache.filename = os.path.join(umls_tables, 'umls_cache')
            UmlsCache.cache = load_pickled_obj(UmlsCache.filename)

        except IOError:
            UmlsCache.cache = {}

    def has_key(self, string):
        return UmlsCache.cache.has_key(string)

    def add_map(self, string, mapping):
        UmlsCache.cache[string] = mapping

    def get_map(self, string):
        return UmlsCache.cache[string]

    @staticmethod
    @atexit.register
    def destructor():

        if UmlsCache.filename is not None and UmlsCache.cache is not None:
            pickle.dump(UmlsCache.cache, open(UmlsCache.filename, "wb"))
