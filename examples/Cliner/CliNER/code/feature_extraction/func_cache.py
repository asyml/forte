"""
 File Name : func_cache.py

 Creation Date : 19-09-2015
<<<<<<< HEAD
                                                                              
 Last Modified : Sun 25 Oct 2015 05:15:04 PM EDT
                                                                              
 Created By : Renan Campos                                                    
                                                                              
 Purpose : Class wrapper for third-party lru cache. 
           Adds method used for reporting hits/misses.                                                                 
=======

 Last Modified : Wed 07 Oct 2015 09:06:27 PM EDT

 Created By : Renan Campos

 Purpose : Class wrapper for third-party lru cache.
           Adds method used for reporting hits/misses.
>>>>>>> 6403f56bb687b79483093ed2a3281df607b690ce
"""

from repoze.lru import lru_cache

class func_cache(lru_cache):

    def __init__(self, verbose=False):
        super(func_cache, self).__init__(500)
        self.verbose = verbose

    def ShowInfo(self):
        # This function is only implicitly called if verbose flag is set.
        print "Cache results for:", self.FuncName
        print "   hits:", self.cache.hits
        print "   misses:", self.cache.misses
        print "   lookups:", self.cache.lookups, "\n"
    
    def __call__(self, f):
        lru_cached = super(func_cache, self).__call__(f)
        lru_cached.ShowInfo = self.ShowInfo
        self.FuncName = f.__name__
        return lru_cached

    def __del__(self):
        if (self.verbose):
            self.ShowInfo()

# Test functionality
if __name__ == '__main__':
    @func_cache()
    def rec(n):
        if not n:
            return n
        return rec(n-1)

    rec.ShowInfo()
    print
    rec(3)
    rec.ShowInfo()
    print
    rec(3)
    rec.ShowInfo()

