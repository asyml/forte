if __name__ == '__main__':
    import doctest

    import os
    import sys

    home = os.path.join(os.getenv('CLINER_DIR'), 'cliner')
    if home not in sys.path:
        sys.path.append(home)

    # from features_dir import *

    import features_dir.features

    doctest.testmod(features_dir.features)

    import features_dir.read_config

    doctest.testmod(features_dir.read_config)

    import features_dir.sentence_features

    doctest.testmod(features_dir.sentence_features)

    import features_dir.utilities

    doctest.testmod(features_dir.utilities)

    import features_dir.word_features

    doctest.testmod(features_dir.word_features)
