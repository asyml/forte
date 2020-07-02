# copied from: https://github.com/exogen/text2num/blob/289745aebaf91e312fa8f8d86e04c17d7a3771af/text2num.py

# This library is a simple implementation of a function to convert textual
# numbers written in English into their integer representations.
#
# This code is open source according to the MIT License as follows.
#
# Copyright (c) 2008 Greg Hewgill
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""
Convert textual numbers written in English into their integer representations.

>>> text2num("zero")
0

>>> text2num("one")
1

>>> text2num("twelve")
12

>>> text2num("nineteen")
19

>>> text2num("twenty nine")
29

>>> text2num("seventy two")
72

>>> text2num("three hundred")
300

>>> text2num("twelve hundred")
1200

>>> text2num("nineteen hundred eighty four")
1984


Hundreds may be implied without a 'hundreds' token if no other magnitudes
are present:

>>> text2num("one thirty")
130

>>> text2num("six sixty two")
662

>>> text2num("ten twelve")
1012

>>> text2num("nineteen ten")
1910

>>> text2num("nineteen eighty four")
1984

>>> text2num("twenty ten")
2010

>>> text2num("twenty twenty")
2020

>>> text2num("twenty twenty one")
2021

>>> text2num("fifty sixty three")
5063

>>> text2num("one thirty thousand")
Traceback (most recent call last):
    ...
NumberException: 'thousand' may not proceed implied hundred 'one thirty'

>>> text2num("nineteen eighty thousand")
Traceback (most recent call last):
    ...
NumberException: 'thousand' may not proceed implied hundred
'nineteen eighty'

>>> text2num("twelve thousand three hundred four")
12304

>>> text2num("six million")
6000000

>>> text2num("six million four hundred thousand five")
6400005

>>> text2num("one hundred twenty three billion four hundred fifty six "
...          "million seven hundred eighty nine thousand twelve")
123456789012

>>> text2num("four decillion")
4000000000000000000000000000000000

>>> text2num("one hundred thousand")
100000

>>> text2num("one hundred two thousand")
102000


Magnitudes must magnify a number and appear in descending order (except
for hundreds, since it can magnify other magnitudes).

>>> text2num("thousand")
Traceback (most recent call last):
    ...
NumberException: magnitude 'thousand' must be preceded by a number

>>> text2num("hundred one")
Traceback (most recent call last):
    ...
NumberException: magnitude 'hundred' must be preceded by a number

>>> text2num("one thousand thousand")
Traceback (most recent call last):
    ...
NumberException: magnitude 'thousand' must be preceded by a number

>>> text2num("one thousand two thousand")
Traceback (most recent call last):
    ...
NumberException: magnitude 'thousand' appeared out of order following
'one thousand two'

>>> text2num("one hundred two hundred")
Traceback (most recent call last):
    ...
NumberException: magnitude 'hundred' appeared out of order following
'one hundred two'

>>> text2num("one thousand two million")
Traceback (most recent call last):
    ...
NumberException: magnitude 'million' appeared out of order following
'one thousand two'

>>> text2num("nine one")
Traceback (most recent call last):
    ...
NumberException: 'one' may not proceed 'nine'

>>> text2num("ten two")
Traceback (most recent call last):
    ...
NumberException: 'two' may not proceed 'ten'

>>> text2num("nineteen nine")
Traceback (most recent call last):
    ...
NumberException: 'nine' may not proceed 'nineteen'

>>> text2num("sixty five hundred")
6500

>>> text2num("sixty hundred")
6000

>>> text2num("ten hundred twelve")
1012

>>> text2num("twenty twenty ten")
Traceback (most recent call last):
    ...
NumberException: 'ten' may not proceed 'twenty' following 'twenty'

>>> text2num("three thousand nineteen eighty four")
Traceback (most recent call last):
    ...
NumberException: 'eighty' may not proceed 'nineteen' following
'three thousand'

>>> text2num("three million nineteen eighty four")
Traceback (most recent call last):
    ...
NumberException: 'eighty' may not proceed 'nineteen' following
'three million'

>>> text2num("one million eighty eighty")
Traceback (most recent call last):
    ...
NumberException: 'eighty' may not proceed 'eighty' following 'one million'

>>> text2num("one million eighty one")
1000081

>>> text2num("zero zero")
Traceback (most recent call last):
    ...
NumberException: 'zero' may not appear with other numbers

>>> text2num("one zero")
Traceback (most recent call last):
    ...
NumberException: 'zero' may not appear with other numbers

>>> text2num("zero thousand")
Traceback (most recent call last):
    ...
NumberException: 'zero' may not appear with other numbers

>>> text2num("foo thousand")
Traceback (most recent call last):
    ...
NumberException: unknown number: 'foo'


Strings may optionally include the word 'and', but only in positions
that make sense:

>>> text2num("one thousand and two")
1002

>>> text2num("ten hundred and twelve")
1012

>>> text2num("nineteen hundred and eighty eight")
1988

>>> text2num("one hundred and ten thousand and one")
110001

>>> text2num("forty and two")
Traceback (most recent call last):
    ...
NumberException: 'and' must be preceeded by a magnitude but got 'forty'

>>> text2num("one and")
Traceback (most recent call last):
    ...
NumberException: 'and' must be preceeded by a magnitude but got 'one'

>>> text2num("and one")
Traceback (most recent call last):
    ...
NumberException: 'and' must be preceeded by a magnitude

>>> text2num("one hundred and")
Traceback (most recent call last):
    ...
NumberException: 'and' must be followed by a number

>>> text2num("nineteen and eighty eight")
Traceback (most recent call last):
    ...
NumberException: 'and' must be preceeded by a magnitude but got 'nineteen'

"""

import re

SMALL = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90
}

MAGNITUDE = {
    'hundred':      100,
    'thousand':     1000,
    'million':      1000000,
    'billion':      1000000000,
    'trillion':     1000000000000,
    'quadrillion':  1000000000000000,
    'quintillion':  1000000000000000000,
    'sextillion':   1000000000000000000000,
    'septillion':   1000000000000000000000000,
    'octillion':    1000000000000000000000000000,
    'nonillion':    1000000000000000000000000000000,
    'decillion':    1000000000000000000000000000000000,
}


class NumberException(Exception):
    """
    Number parsing error.

    """
    pass


def text2num(s):
    """
    Convert the English number phrase `s` into the integer it describes.

    """
    # pylint: disable=invalid-name,too-many-branches,undefined-loop-variable
    words = re.split(r'[\s,-]+', s)

    if not words:
        raise NumberException("no numbers in string: {!r}".format(s))

    n = 0
    g = 0
    implied_hundred = False

    for i, word in enumerate(words):
        tens = g % 100
        if word == "and":
            if i and tens == 0:
                # If this isn't the first word, and `g` was multiplied by 100
                # or reset to 0, then we're in a spot where 'and' is allowed.
                continue
            else:
                fmt = (word, " but got {!r}".format(words[i - 1]) if i else "")
                raise NumberException("{!r} must be preceeded by a magnitude"
                                      "{}".format(*fmt))

        x = SMALL.get(word, None)
        if x is not None:
            if x == 0 and len(words) > 1:
                raise NumberException("{!r} may not appear with other "
                                      "numbers".format(word))

            if tens != 0:
                # Check whether the two small numbers can be treated as if an
                # implied 'hundred' is present, as in 'nineteen eighty four'.
                if x >= 10:
                    # Only allow implied hundreds if no other magnitude is
                    # already present.
                    if n == 0:
                        n += g * 100
                        g = 0
                        implied_hundred = True
                    else:
                        fmt = (word, words[i - 1], " ".join(words[:i - 1]))
                        raise NumberException("{!r} may not proceed {!r} "
                                              "following {!r}".format(*fmt))
                # Treat sequences like 'nineteen one' as errors rather than
                # interpret them as 'nineteen hundred one', 'nineteen aught
                # one', 'nineteen oh one', etc. But continue if we have 20 or
                # greater in the accumulator to support 'twenty one', 'twenty
                # two', etc.
                elif tens < 20:
                    raise NumberException("{!r} may not proceed "
                                          "{!r}".format(word, words[i - 1]))

            g += x
        else:
            x = MAGNITUDE.get(word, None)
            if x is None:
                raise NumberException("unknown number: {!r}".format(word))
            # We could check some of these branches in the conditional one
            # level up, but would prefer the 'unknown number' exception take
            # precedence since it's a bigger problem.
            elif implied_hundred:
                fmt = (word, " ".join(words[:i]))
                raise NumberException("{!r} may not proceed implied hundred "
                                      "{!r}".format(*fmt))
            # Disallow standalone magnitudes and multiple magnitudes like
            # 'one thousand million' where 'one billion' should be used
            # instead.
            elif g == 0:
                raise NumberException("magnitude {!r} must be preceded by a "
                                      "number".format(word))
            # Check whether this magnitude was preceded by a lower one.
            elif 0 < n <= x or g >= x:
                fmt = (word, " ".join(words[:i]))
                raise NumberException("magnitude {!r} appeared out of order "
                                      "following {!r}".format(*fmt))
            # Accumulate hundreds in `g`, not `n`, since hundreds can magnify
            # other magnitudes.
            elif x == 100:
                g *= x
            else:
                n += g * x
                g = 0

    # We could check whether the last word is 'and' at the very beginning and
    # fail early, but this way errors are raised in the order each word is
    # seen, as if we're processing a stream.
    if word == "and":
        raise NumberException("{!r} must be followed by a number".format(word))

    return n + g
