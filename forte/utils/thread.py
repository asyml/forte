# Copyright 2019 The Forte Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading


class AtomicCounter:
    """ An Atomic Counter that is supposed to be thread safe. Code adopted
    from:
    https://gist.github.com/benhoyt/8c8a8d62debe8e5aa5340373f9c509c7

    >>> counter = AtomicCounter()
    >>> counter.value
    0
    >>> counter.increment()
    1
    >>> counter.increment(2)
    3


    >>> counter = AtomicCounter()
    >>> def incrementor():
    ...     for i in range(100000):
    ...         counter.increment()
    >>> threads = []
    >>> for i in range(5):
    ...     thread = threading.Thread(target=incrementor())
    ...     thread.start()
    ...     threads.append(thread)
    >>> for thread in threads:
    ...     thread.join()
    >>> counter.value
    500000

    """

    def __init__(self, initial=0):
        self.value = initial
        self.__lock = threading.Lock()

    def increment(self, amount=1) -> int:
        with self.__lock:
            self.value += amount
            return self.value


if __name__ == "__main__":
    import doctest
    doctest.testmod()
