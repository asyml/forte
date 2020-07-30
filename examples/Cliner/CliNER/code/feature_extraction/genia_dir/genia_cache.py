import os
import pickle
import sys

from .. import utils

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parentdir not in sys.path:
    sys.path.append(parentdir)


class GeniaCache:
    def __init__(self):
        try:
            prefix = os.path.dirname(__file__)
            self.filename = os.path.join(prefix, 'genia_cache')
            self.cache = utils.load_pickled_obj(self.filename)
        except IOError:
            self.cache = {}

    def has_key(self, key):
        return str(key) in self.cache

    def add_map(self, key, value):
        self.cache[str(key)] = value

    def get_map(self, key):
        return self.cache[str(key)]

    def __del__(self):
        pickle.dump(self.cache, open(self.filename, "wb"))
