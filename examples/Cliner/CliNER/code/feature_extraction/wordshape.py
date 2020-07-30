import re

BOUNDARY_SIZE = 2

NOWORDSHAPE = -1
WORDSHAPEDAN1 = 0
WORDSHAPECHRIS1 = 1
WORDSHAPEDAN2 = 2
WORDSHAPEDAN2USELC = 3
WORDSHAPEDAN2BIO = 4
WORDSHAPEDAN2BIOUSELC = 5
WORDSHAPEJENNY1 = 6
WORDSHAPEJENNY1USELC = 7
WORDSHAPECHRIS2 = 8
WORDSHAPECHRIS2USELC = 9
WORDSHAPECHRIS3 = 10
WORDSHAPECHRIS3USELC = 11

greek = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "theta", "iota",
         "kappa", "lambda", "omicron", "rho",
         "sigma", "tau", "upsilon", "omega"]
biogreek = r"alpha|beta|gamma|delta|epsilon|zeta|theta|iota|kappa|lambda" \
           r"|omicron|rho|sigma|tau|upsilon|omega"


def lookupShaper(name):
    if name is None:
        return NOWORDSHAPE
    elif name.lower() == "dan1":
        return WORDSHAPEDAN1
    elif name.lower() == "chris1":
        return WORDSHAPECHRIS1
    elif name.lower() == "dan2":
        return WORDSHAPEDAN2
    elif name.lower() == "dan2useLC":
        return WORDSHAPEDAN2USELC
    elif name.lower() == "dan2bio":
        return WORDSHAPEDAN2BIO
    elif name.lower() == "dan2bioUseLC":
        return WORDSHAPEDAN2BIOUSELC
    elif name.lower() == "jenny1":
        return WORDSHAPEJENNY1
    elif name.lower() == "jenny1useLC":
        return WORDSHAPEJENNY1USELC
    elif name.lower() == "chris2":
        return WORDSHAPECHRIS2
    elif name.lower() == "chris2useLC":
        return WORDSHAPECHRIS2USELC
    elif name.lower() == "chris3":
        return WORDSHAPECHRIS3
    elif name.lower() == "chris3useLC":
        return WORDSHAPECHRIS3USELC
    else:
        return NOWORDSHAPE


def dontUseLC(shape):
    return shape in (
        WORDSHAPEDAN2, WORDSHAPEDAN2BIO, WORDSHAPEJENNY1, WORDSHAPECHRIS2,
        WORDSHAPECHRIS3)


def wordShapeNext(inStr, wordShaper, knownLCWords):
    if knownLCWords is not None and dontUseLC(wordShaper):
        knownLCWords = None

    if wordShaper == NOWORDSHAPE:
        return inStr
    elif wordShaper == WORDSHAPEDAN1:
        return wordShapeDan1(inStr)
    elif wordShaper == WORDSHAPECHRIS1:
        return wordShapeChris1(inStr)
    elif wordShaper == WORDSHAPEDAN2:
        return wordShapeDan2(inStr, knownLCWords)
    elif wordShaper == WORDSHAPEDAN2USELC:
        return wordShapeDan2(inStr, knownLCWords)
    elif wordShaper == WORDSHAPEDAN2BIO:
        return wordShapeDan2Bio(inStr, knownLCWords)
    elif wordShaper == WORDSHAPEDAN2BIOUSELC:
        return wordShapeDan2Bio(inStr, knownLCWords)
    elif wordShaper == WORDSHAPEJENNY1:
        return wordShapeJenny1(inStr)
    elif wordShaper == WORDSHAPEJENNY1USELC:
        return wordShapeJenny1(inStr)
    elif wordShaper == WORDSHAPECHRIS2:
        return wordShapeChris2(inStr, False, knownLCWords)
    elif wordShaper == WORDSHAPECHRIS2USELC:
        return wordShapeChris2(inStr, False, knownLCWords)
    elif wordShaper == WORDSHAPECHRIS3:
        return wordShapeChris2(inStr, True, knownLCWords)
    elif wordShaper == WORDSHAPECHRIS3USELC:
        return wordShapeChris2(inStr, True, knownLCWords)


def wordShape(inStr, wordShaper):
    return wordShapeNext(inStr, wordShaper, None)


def wordShapeDan1(s):
    digit = True
    upper = True
    lower = True
    mixed = True
    i = 0
    for c in s:
        if not c.isdigit():
            digit = False
        if not c.islower():
            lower = False
        if not c.isupper():
            upper = False
        if (i == 0 and not c.isupper()) or (i >= 1 and not c.islower()):
            mixed = False
        i += 1
    if digit:
        return "ALL-DIGITS"
    if upper:
        return "ALL-UPPER"
    if lower:
        return "ALL-LOWER"
    if mixed:
        return "MIXED-CASE"
    return "OTHER"


def wordShapeDan2(s, knownLCWords):
    sb = "WT-"
    lastM = '~'
    nonLetters = False
    length = len(s)
    for c in s:
        m = c
        if c.isdigit():
            m = 'd'
        if c.islower() or c == '_':
            m = 'x'
        if c.isupper():
            m = 'X'
        # pylint: disable=consider-using-in
        if m != 'x' and m != 'X':
            nonLetters = True
        if m != lastM:
            sb += m
        lastM = m

    if length <= 3:
        sb += ':' + str(length)

    if knownLCWords is not None:
        if not nonLetters and knownLCWords.contains(s.lower()):
            sb += 'k'
    return sb


def wordShapeJenny1(s):
    sb = "WT-"
    lastM = '~'
    length = len(s)

    for i in range(0, length):
        c = s[i]
        m = c
        if c.isdigit():
            m = 'd'
        if c.islower():
            m = 'x'
        if c.isupper():
            m = 'X'

        for gr in greek:
            if s.startswith(gr):
                m = 'g'
                i = i + len(gr) - 1
                break

        if m != lastM:
            sb += m

        lastM = m

    if length <= 3:
        sb += ':' + str(length)

    # if knownLCWords is not None:
    #   if not nonLetters and knownLCWords.contains(s.lower()):
    #      sb += 'k'
    return sb


def wordShapeChris2(s, omitIfInBoundary, knownLCWords):
    length = len(s)
    if length <= BOUNDARY_SIZE * 2:
        return wordShapeChris2Short(s, length, knownLCWords)
    else:
        return wordShapeChris2Long(s, omitIfInBoundary, length, knownLCWords)


def wordShapeChris2Short(s, length, knownLCWords):
    sb = ""

    nonLetters = False

    for i in range(0, length):
        c = s[i]
        m = c
        if c.isdigit():
            m = 'd'
        if c.islower():
            m = 'x'
        if c.isupper() or c.istitle():
            m = 'X'

        for gr in greek:
            if s.startswith(gr):
                m = 'g'
                i = i + len(gr) - 1
                break

        # pylint: disable=consider-using-in
        if m != 'x' and m != 'X':
            nonLetters = True

        sb += m
    # pylint: disable=consider-using-in
    if knownLCWords is not None:
        if not nonLetters and knownLCWords.contains(s.lower()):
            sb += 'k'
    return sb


def wordShapeChris2Long(s, omitIfInBoundary, length, knownLCWords):
    beginChars = ""
    endChars = ""
    beginUpto = 0
    endUpto = 0
    seenSet = set([])

    nonLetters = False
    for _, i in enumerate(range(0, len(s))):
        iIncr = 0
        c = s[i]
        m = c
        if c.isdigit():
            m = 'd'
        elif c.islower():
            m = 'x'
        elif c.isupper() or c.istitle():
            m = 'X'

        for gr in greek:
            if s.startswith(gr):
                m = 'g'
                iIncr = len(gr) - 1
                break
        # pylint: disable=consider-using-in
        if m != 'x' and m != 'X':
            nonLetters = True

        if i < BOUNDARY_SIZE:
            beginChars += m
            beginUpto += 1
        elif i < length - BOUNDARY_SIZE:
            seenSet.add(m)
        else:
            endChars += m
            endUpto += 1
        i += iIncr

        sbSize = beginUpto + endUpto + len(seenSet)

        if knownLCWords is not None:
            sbSize += 1

        sb = ""
        sb += beginChars
        if omitIfInBoundary:
            for ch in seenSet:
                insert = True
                for k in range(0, beginUpto):
                    if beginChars[k] == ch:
                        insert = False
                        break

                for k in range(0, endUpto):
                    if endChars[k] == ch:
                        insert = False
                        break
                if insert:
                    sb += ch
        else:
            for ch in seenSet:
                sb += ch
    sb += endChars
    if knownLCWords is not None:
        if not nonLetters and knownLCWords.contains(s.tolower()):
            sb += 'k'
    return sb


def wordShapeDan2Bio(s, knownLCWords):
    if containsGreekLetter(s):
        return wordShapeDan2(s, knownLCWords) + "-GREEK"
    else:
        return wordShapeDan2(s, knownLCWords)


def containsGreekLetter(s):
    return re.search(biogreek, s)


def wordShapeChris1(s):
    length = len(s)
    if length == 0:
        return "SYMBOL"

    cardinal = False
    number = True
    seenDigit = False
    seenNonDigit = False

    for i in range(0, length):
        ch = s[i]
        digit = ch.isdigit()
        if digit:
            seenDigit = True
        else:
            seenNonDigit = True
        # pylint: disable=consider-using-in
        digit = digit or ch == '.' or ch == ',' or (
                i == 0 and (ch == '-' or ch == '+'))
        if not digit:
            number = False

    if not seenDigit:
        number = False
    elif not seenNonDigit:
        cardinal = True

    if cardinal:
        if length < 4:
            return "CARDINAL13"
        elif length == 4:
            return "CARDINAL4"
        else:
            return "CARDINAL5PLUS"
    elif number:
        return "NUMBER"

    seenLower = False
    seenUpper = False
    allCaps = True
    allLower = True
    initCap = False
    dash = False
    period = False

    for i in range(0, length):
        ch = s[i]
        up = ch.isupper()
        let = re.search(r"^[A-Za-z]+$", ch)
        tit = ch.istitle()
        if ch == '-':
            dash = True
        elif ch == '.':
            period = True

        if tit:
            seenUpper = True
            allLower = False
            seenLower = True
            allCaps = False
        elif up:
            seenUpper = True
            allLower = False
        elif let:
            seenLower = True
            allCaps = False

        if i == 0 and (up or tit):
            initCap = True

    if length == 2 and initCap and period:
        return "ACRONYM1"
    elif seenUpper and allCaps and not seenDigit and period:
        return "ACRONYM"
    elif seenDigit and dash and not seenUpper and not seenLower:
        return "DIGIT-DASH"
    elif initCap and seenLower and seenDigit and dash:
        return "CAPITALIZED-DIGIT-DASH"
    elif initCap and seenLower and seenDigit:
        return "CAPITALIZED-DIGIT"
    elif initCap and seenLower & dash:
        return "CAPITALIZED-DASH"
    elif initCap and seenLower:
        return "CAPITALIZED"
    elif seenUpper and allCaps and seenDigit and dash:
        return "ALLCAPS-DIGIT-DASH"
    elif seenUpper and allCaps and seenDigit:
        return "ALLCAPS-DIGIT"
    elif seenUpper and allCaps and dash:
        return "ALLCAPS"
    elif seenUpper and allCaps:
        return "ALLCAPS"
    elif seenLower and allLower and seenDigit and dash:
        return "LOWERif wordShaper ==-DIGIT-DASH"
    elif seenLower and allLower and seenDigit:
        return "LOWERif wordShaper ==-DIGIT"
    elif seenLower and allLower and dash:
        return "LOWERif wordShaper ==-DASH"
    elif seenLower and allLower:
        return "LOWERif wordShaper =="
    elif seenLower and seenDigit:
        return "MIXEDif wordShaper ==-DIGIT"
    elif seenLower:
        return "MIXEDif wordShaper =="
    elif seenDigit:
        return "SYMBOL-DIGIT"
    else:
        return "SYMBOL"


# gets Chris1, Dan1, Jenny1, Chris2 and Dan2 word shapes
def getWordShapes(word):
    return [wordShapeChris1(word), wordShapeDan1(word), wordShapeJenny1(word),
            wordShapeChris2(word, False, None),
            wordShapeDan2(word, None)]
