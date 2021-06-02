# Idea from https://stackoverflow.com/questions/33508060/create-and-import-helper-functions-in-tests-without-creating-packages-in-test-di

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))
