from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import json
import os
import random
import math
import argparse
from collections import Counter, defaultdict, namedtuple
from typing import Dict, List, Set, Any

from nltk import sent_tokenize, word_tokenize
import numpy as np
import h5py

# pylint: disable=unused-variable,redefined-outer-name,unused-argument
# pylint: disable=dangerous-default-value,unnecessary-lambda, cell-var-from-loop
# pylint: disable=undefined-loop-variable,expression-not-assigned
# pylint: disable=consider-using-enumerate


from examples.content_rewriter.model.data2text.text2num import (
    text2num, NumberException)


def divide_or_const(a, b, c=0.):
    try:
        return a / b
    except ZeroDivisionError:
        return c


random.seed(2)

Ent = namedtuple("Ent", ["start", "end", "s", "is_pron"])
Num = namedtuple("Num", ["start", "end", "s"])
Rel = namedtuple("Rel", ["ent", "num", "type", "aux"])
stuff_names = ('sent', 'len', 'entdist', 'numdist', 'label')

prons = {"he", "He", "him", "Him", "his", "His", "they", "They",
         "them", "Them", "their", "Their"}  # leave out "it"
singular_prons = {"he", "He", "him", "Him", "his", "His"}
plural_prons = {"they", "They", "them", "Them", "their", "Their"}

number_words = {"one", "two", "three", "four", "five", "six", "seven", "eight",
                "nine", "ten", "eleven", "twelve",
                "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                "eighteen", "nineteen", "twenty", "thirty",
                "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
                "hundred", "thousand"}

line_score_words: Dict[str, List[str]] = {
    'TEAM-AST': ['assist'],
    'TEAM-FG3_PCT': ['percent'],
    'TEAM-FG_PCT': ['percent'],
    'TEAM-FT_PCT': ['percent'],
    'TEAM_LOSSES': [],
    'TEAM-PTS': ['point'],
    'TEAM-PTS_QTR1': ['point'],
    'TEAM-PTS_QTR2': ['point'],
    'TEAM-PTS_QTR3': ['point'],
    'TEAM-PTS_QTR4': ['point'],
    'TEAM-REB': ['rebound'],
    'TEAM-TOV': ['turnover'],
    'TEAM-WINS': [],
}

box_score_words = {
    'AST': ['assist'],
    'BLK': ['block'],
    'DREB': ['rebound'],
    'FG3A': [],
    'FG3M': [],
    'FG3_PCT': ['percent'],
    'FGA': [],
    'FGM': [],
    'FG_PCT': ['percent'],
    'FTA': [],
    'FTM': [],
    'FT_PCT': ['percent'],
    'MIN': ['minute'],
    'OREB': ['rebound'],
    'PF': ['foul'],
    'PTS': ['point'],
    'REB': ['rebound'],
    'STL': ['steal'],
    'TO': ['turnover'],
}
box_score_words: Dict[str, Any] = {
    'PLAYER-{}'.format(name): value for name, value in
    box_score_words.items()}

score_words: Dict[str, List[str]] = line_score_words.copy()
score_words.update(box_score_words)

indicating_words: Set[str] = set()
for words in score_words.values():
    indicating_words.update(words)


def starts_with_indicating_word(token):
    for word in indicating_words:
        if token.startswith(word):
            return word
    return None


def get_json_dataset(path, stage):
    with open(os.path.join(path, "{}.json".format(stage)), 'r') as json_file:
        return json.load(json_file)


def get_ents(dat):
    players = set()
    teams = set()
    cities = set()
    team_strs = ["vis", "home"]
    for thing in dat:
        for team_str in team_strs:
            names = thing["{}_name".format(team_str)], \
                    thing["{}_line".format(team_str)]["TEAM-NAME"]
            prefixes = ["", thing["{}_city".format(team_str)] + " "]
            for prefix in prefixes:
                for name in names:
                    teams.add(prefix + name)
            # special case for this
            if thing["{}_city".format(team_str)] == "Los Angeles":
                teams.add("LA" + thing["{}_name".format(team_str)])
            # sometimes team_city is different
            cities.add(thing["{}_city".format(team_str)])
        players.update(thing["box_score"]["PLAYER_NAME"].values())
        cities.update(thing["box_score"]["TEAM_CITY"].values())

    for entset in [players, teams, cities]:
        for k in list(entset):
            pieces = k.split()
            if len(pieces) > 1:
                for piece in pieces:
                    if len(piece) > 1 and piece not in ["II", "III", "Jr.",
                                                        "Jr"]:
                        entset.add(piece)

    all_ents = players | teams | cities

    return all_ents, players, teams, cities


def get_train_ents(path="rotowire", connect_multiwords=False):
    datasets = {}
    for stage in ["train"]:
        with open(os.path.join(path, "{}.json".format(stage)), "r") as f:
            datasets[stage] = json.load(f)
    list_of_ents = get_ents(datasets["train"])
    if connect_multiwords:
        list_of_ents = tuple(
            set(map(lambda s: s.replace(' ', '_'), ents))
            for ents in list_of_ents)
    return list_of_ents


def deterministic_resolve(pron, players, teams, cities, curr_ents, prev_ents,
                          max_back=1):
    # we'll just take closest compatible one.
    # first look in current sentence; if there's an antecedent here return None,
    # since we'll catch it anyway
    for j in range(len(curr_ents) - 1, -1, -1):
        if pron in singular_prons and curr_ents[j][2] in players:
            return None
        elif pron in plural_prons and curr_ents[j][2] in teams:
            return None
        elif pron in plural_prons and curr_ents[j][2] in cities:
            return None

    # then look in previous max_back sentences
    if len(prev_ents) > 0:
        for i in range(len(prev_ents) - 1, len(prev_ents) - 1 - max_back, -1):
            for j in range(len(prev_ents[i]) - 1, -1, -1):
                if pron in singular_prons and prev_ents[i][j][2] in players:
                    return prev_ents[i][j]
                elif pron in plural_prons and prev_ents[i][j][2] in teams:
                    return prev_ents[i][j]
                elif pron in plural_prons and prev_ents[i][j][2] in cities:
                    return prev_ents[i][j]
    return None


def extract_entities(sent, all_ents, prons=prons, prev_ents=None,
                     resolve_prons=False,
                     players=None, teams=None, cities=None):
    sent_ents = []
    i = 0
    while i < len(sent):
        if sent[i] in prons:
            if resolve_prons:
                referent = deterministic_resolve(sent[i], players, teams,
                                                 cities, sent_ents, prev_ents)
                if referent is None:
                    sent_ents.append(
                        Ent(i, i + 1, sent[i], True))  # is a pronoun
                else:
                    # pretend it's not a pron and put in matching string
                    sent_ents.append(
                        Ent(i, i + 1, referent[2], False))
            else:
                sent_ents.append(Ent(i, i + 1, sent[i], True))  # is a pronoun
            i += 1
        elif sent[
            # findest longest spans; only works if we put in words...
            i] in all_ents:
            j = 1
            while i + j <= len(sent) and " ".join(sent[i:i + j]) in all_ents:
                j += 1
            sent_ents.append(
                Ent(i, i + j - 1, " ".join(sent[i:i + j - 1]), False))
            i += j - 1
        else:
            i += 1
    return sent_ents


def annoying_number_word(sent, i):
    ignores = [
        ((0, 2), {"three point", "three pt", "three pointers", "one of"}),
        ((0, 3),
         {"three - point", "three - pt", "three - pointer", "three - pointers",
          "three - points"}),
        ((-1, 1), {"this one"}),
    ]
    for span, words in ignores:
        if " ".join(sent[i + span[0]: i + span[1]]) in words:
            return True
    return False


def extract_numbers(sent):
    sent_nums = []
    i = 0
    while i < len(sent):
        j = i + 1
        try:
            n = int(sent[i])
        except ValueError:
            if sent[i] in ["a", "an"] \
                    and i + 1 < len(sent) \
                    and starts_with_indicating_word(sent[i + 1]) is not None:
                n = 1
            elif sent[i] in number_words and not annoying_number_word(sent, i):
                # get longest span  (this is kind of stupid)
                while j < len(sent) \
                        and sent[j] in number_words \
                        and not annoying_number_word(sent, j):
                    j += 1
                try:
                    n = text2num(" ".join(sent[i:j]))
                except NumberException:
                    j = i + 1
                    n = text2num(sent[i])
            else:
                n = None
        if n is not None:
            sent_nums.append(Num(i, j, n))
        i = j
    return sent_nums


def get_player_idx(bs, entname):
    keys = []
    for k, v in bs["PLAYER_NAME"].items():
        if entname == v:
            keys.append(k)
    if len(keys) == 0:
        for k, v in bs["SECOND_NAME"].items():
            if entname == v:
                keys.append(k)
        if len(keys) > 1:  # take the earliest one
            keys.sort(key=lambda x: int(x))
            keys = keys[:1]
            # print("picking", bs["PLAYER_NAME"][keys[0]])
    if len(keys) == 0:
        for k, v in bs["FIRST_NAME"].items():
            if entname == v:
                keys.append(k)
        if len(keys) > 1:
            # if we matched on first name and there are a bunch
            # just forget about it
            return None
    # if len(keys) == 0:
    # print("Couldn't find", entname, "in", bs["PLAYER_NAME"].values())
    assert len(keys) <= 1, entname + " : " + str(bs["PLAYER_NAME"].values())
    return keys[0] if len(keys) > 0 else None


def get_rels(entry, tokens, ents, nums, players, teams, cities,
             filter_none=False):
    """
    this looks at the box/line score and figures out which (entity, number)
    pairs are candidate true relations, and which can't be.
    if an ent and number don't line up (i.e., aren't in the box/line score
    together), we give a NONE label, so for generated summaries that we
    extract from, if we predict a label we'll get it wrong (which is presumably
    what we want). N.B. this function only looks at the entity string (not
    position in sentence), so the string a pronoun corefers with can be
    snuck in....
    """
    rels = []
    bs = entry["box_score"]

    ent_strs = [ent.s.split() for ent in ents]

    def is_complete(ent):
        ent_str = ent.s.split()
        for ent_str_ in ent_strs:
            if len(ent_str_) <= len(ent_str):
                continue
            for idx in range(len(ent_str_) - len(ent_str) + 1):
                if ent_str == ent_str_[idx: idx + len(ent_str)]:
                    return False
        return True

    new_ents = set(filter(is_complete, ents))
    ents = new_ents

    for i, ent in enumerate(ents):
        if ent.is_pron:  # pronoun
            continue  # for now
        entname = ent.s
        # assume if a player has a city or team name as his name,
        # they won't use that one (e.g., Orlando Johnson)
        if entname in players and entname not in cities \
                and entname not in teams:
            pidx = get_player_idx(bs, entname)
            for j, numtup in enumerate(nums):
                found = False
                strnum = str(numtup.s)
                # player might not actually be in the game or whatever
                if pidx is not None:
                    for colname, col in bs.items():
                        if col[pidx] == strnum:  # allow multiple for now
                            rels.append(
                                Rel(ent, numtup, "PLAYER-" + colname, pidx))
                            found = True
                if not found:
                    if not filter_none:
                        rels.append(Rel(ent, numtup, "NONE", None))

        else:  # has to be city or team
            entpieces = entname.split()
            linescore = None
            is_home = None
            if entpieces[0] in entry["home_city"] or entpieces[-1] in entry[
                "home_name"]:
                linescore = entry["home_line"]
                is_home = True
            elif entpieces[0] in entry["vis_city"] or entpieces[-1] in entry[
                "vis_name"]:
                linescore = entry["vis_line"]
                is_home = False
            elif "LA" in entpieces[0]:
                if entry["home_city"] == "Los Angeles":
                    linescore = entry["home_line"]
                    is_home = True
                elif entry["vis_city"] == "Los Angeles":
                    linescore = entry["vis_line"]
                    is_home = False
            for j, numtup in enumerate(nums):
                found = False
                strnum = str(numtup.s)
                if linescore is not None:
                    for colname, val in linescore.items():
                        if val == strnum:
                            # apparently I appended TEAM- at some pt...
                            rels.append(Rel(ent, numtup, colname, is_home))
                            found = True
                if not found:
                    if not filter_none:
                        # should i specialize the NONE labels too?
                        rels.append(Rel(ent, numtup, "NONE", None))

    filt = (lambda cond, rels: filter(cond, rels)) if filter_none else \
        (lambda cond, rels: map(lambda rel: rel if cond(rel) else
        Rel(rel.ent, rel.num, "NONE", None), rels))

    filtered_rels = []
    for num in nums:
        related_rels = list(filter(lambda rel: rel.num == num, rels))
        cnt = len(list(filter(lambda rel: rel.type != "NONE", related_rels)))
        indicating_word = starts_with_indicating_word(tokens[num.end])
        if indicating_word is not None:
            def correct(rel):
                try:
                    return indicating_word in score_words[rel.type]
                except KeyError:
                    return True

            related_rels = list(filt(correct, related_rels))
        filtered_rels.extend(related_rels)
    rels = filtered_rels

    def ensure(num, rel_type, ent=None):
        def correct(rel):
            if rel.num != num:
                return True
            if rel.type != rel_type:
                return False
            if ent is not None and rel.ent != ent:
                return False
            return True

        return correct

    ensurers = []
    for i in range(1, len(tokens) - 2):
        if tokens[i] != '-':
            continue
        ns = []
        for j in [i - 1, i + 1]:
            for num in nums:
                if num.start == j and num.end == j + 1:
                    ns.append(num)
                    break
            else:
                break
        if len(ns) < 2:
            continue
        if tokens[i + 2] == 'FG':
            ensurers.append(ensure(ns[0], 'PLAYER-FGM'))
            ensurers.append(ensure(ns[1], 'PLAYER-FGA'))
        elif tokens[i + 2] == '3Pt':
            ensurers.append(ensure(ns[0], 'PLAYER-FG3M'))
            ensurers.append(ensure(ns[1], 'PLAYER-FG3A'))
        elif tokens[i + 2] == 'FT':
            ensurers.append(ensure(ns[0], 'PLAYER-FTM'))
            ensurers.append(ensure(ns[1], 'PLAYER-FTA'))
        elif tokens[i + 2] == ')' and tokens[i - 2] == '(':
            for ent in ents:
                if ent.end == i - 2:
                    break
            else:
                ent = None
            ensurers.append(ensure(ns[0], 'TEAM-WINS', ent))
            ensurers.append(ensure(ns[1], 'TEAM-LOSSES', ent))
    for ensurer in ensurers:
        rels = filt(ensurer, rels)
    rels = list(rels)

    return rels


def do_connect_multiwords(tokes, rels, ents, nums):
    if not tokes:
        return tokes, rels

    retained = [1 for i in range(len(
        tokes))]  # retained[i] means retaining connection between tokens i, i+1
    for items in [ents, nums]:
        for item in items:
            for i in range(item.start, item.end - 1):  # connect these tokens
                retained[i] = 0

    # process target tokens to connect multiword with underscore
    new_tokens = [tokes[0]]
    for i in range(1, len(tokes)):
        token = tokes[i]
        if retained[i - 1]:
            new_tokens.append(token)
        else:
            new_tokens[-1] = new_tokens[-1] + '_' + token

    new_loc = [0]
    for flag in retained:
        new_loc.append(new_loc[-1] + flag)

    new_rels = []
    for rel in rels:
        ent = rel.ent
        ent = Ent(start=new_loc[ent.start],
                  end=new_loc[ent.end],
                  s=ent.s.replace(' ', '_'),
                  is_pron=ent.is_pron)
        assert ent.end - ent.start == 1
        assert new_tokens[
                   ent.start] == ent.s, "new_tokens = {}, ent = {}".format(
            new_tokens, ent)
        num = rel.num
        num = Num(start=new_loc[num.start],
                  end=new_loc[num.end],
                  s=num.s)
        assert num.end - num.start == 1
        new_rels.append(Rel(ent, num, rel.type, rel.aux))

    return new_tokens, new_rels


def get_candidate_rels(entry, summ, all_ents, prons, players, teams, cities,
                       connect_multiwords=False, filter_none=False):
    """
    generate tuples of form (sentence_tokens, [rels]) to candrels
    """
    sents = sent_tokenize(summ)
    for j, sent in enumerate(sents):
        # tokes = word_tokenize(sent)
        tokes = sent.split()
        ents = extract_entities(tokes, all_ents, prons)
        nums = extract_numbers(tokes)
        rels = list(get_rels(entry, tokes, ents, nums, players, teams, cities,
                             filter_none=filter_none))
        if connect_multiwords:
            tokes, rels = do_connect_multiwords(tokes, rels, ents, nums)
        if len(rels) > 0:
            yield (tokes, rels)


stages = ["train", "valid", "test"]


def get_to_data(tup, vocab, labeldict, max_len):
    """
    tup is (sent, [rels]);
    each rel is
    ((ent_start, ent_ent, ent_str), (num_start, num_end, num_str), label)
    """
    sent = [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in tup[0]]
    sentlen = len(sent)
    sent.extend([-1] * (max_len - sentlen))
    for rel in tup[1]:
        ent, num, label, idthing = rel
        ent_dists = [
            j - ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0
            for j in range(max_len)]
        num_dists = [
            j - num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0
            for j in range(max_len)]
        yield sent, sentlen, ent_dists, num_dists, labeldict[label]


def get_multilabeled_data(tup, vocab, labeldict, max_len):
    """
    used for val, since we have contradictory labelings...
    tup is (sent, [rels]);
    each rel is
    ((ent_start, ent_end, ent_str), (num_start, num_end, num_str), label)
    """
    sent = [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in tup[0]]
    sentlen = len(sent)
    sent.extend([-1] * (max_len - sentlen))
    # get all the labels for the same rel
    unique_rels = defaultdict(list)
    for rel in tup[1]:
        ent, num, label, idthing = rel
        unique_rels[ent, num].append(label)

    for rel, label_list in unique_rels.items():
        ent, num = rel
        ent_dists = [
            j - ent[0] if j < ent[0] else j - ent[1] + 1 if j >= ent[1] else 0
            for j in range(max_len)]
        num_dists = [
            j - num[0] if j < num[0] else j - num[1] + 1 if j >= num[1] else 0
            for j in range(max_len)]
        yield sent, sentlen, ent_dists, num_dists, [labeldict[label] for label
                                                    in label_list]


def append_labelnums(labels):
    max_num_labels = max(map(len, labels))
    print("max num labels", max_num_labels)

    # append number of labels to labels
    for i, labellist in enumerate(labels):
        l = len(labellist)
        labellist.extend([-1] * (max_num_labels - l))
        labellist.append(l)


def clean_text(text):
    text = text.replace("\u2019", "'")
    text = text.replace("s ' ", "s 's ")
    text = text.replace("'s ", " 's ")
    text = text.replace("s' ", "s 's ")
    return text


def candidate_rels_extractor(all_ents, players, teams, cities, prons,
                             connect_multiwords=False):
    def extract(entry, summ, filter_none=False):
        return get_candidate_rels(entry, summ, all_ents, prons, players, teams,
                                  cities, connect_multiwords=connect_multiwords)

    return extract


def get_datasets(path, connect_multiwords=False, filter_none=False):
    datasets = {stage: get_json_dataset(path, stage) for stage in stages}
    all_ents, players, teams, cities = get_ents(datasets["train"])
    extract = candidate_rels_extractor(all_ents, players, teams, cities, prons,
                                       connect_multiwords=connect_multiwords)

    extracted_stuff = {}
    for stage, dataset in datasets.items():
        nugz = []
        for i, entry in enumerate(dataset):
            summ = clean_text(" ".join(entry['summary']))
            nugz.extend(extract(entry, summ, filter_none=filter_none))
        nugz = list(
            filter(lambda data: len(data[0]) <= 50 and len(data[1]) <= 50,
                   nugz))
        extracted_stuff[stage] = nugz

    if connect_multiwords:
        all_ents, players, teams, cities = (
            set(map(lambda s: s.replace(' ', '_'), ents))
            for ents in (all_ents, players, teams, cities))

    return extracted_stuff, all_ents, players, teams, cities


# modified full sentence IE training
def save_full_sent_data(outfile, path, multilabel_train=False, nonedenom=0,
                        backup=False, verbose=True):
    datasets = \
        get_datasets(path, connect_multiwords=not backup, filter_none=False)[0]
    # make vocab and get labels
    word_counter = Counter()
    [word_counter.update(tup[0]) for tup in datasets['train']]
    word_counter = Counter(
        {word: cnt for word, cnt in word_counter.items() if cnt >= 2})
    word_counter["UNK"] = 1
    vocab = {wrd: i + 1 for i, wrd in enumerate(sorted(word_counter.keys()))}
    labelset = set()
    [labelset.update(rel.type for rel in tup[1]) for tup in datasets['train']]
    labeldict = {label: i + 1 for i, label in enumerate(sorted(labelset))}

    # save stuff
    stuffs = {stage: [] for stage in datasets}

    max_trlen = max(len(tup[0]) for tup in datasets['train'])
    print("max tr sentence length:", max_trlen)

    # do training data
    for tup in datasets['train']:
        stuffs['train'].extend(
            (get_multilabeled_data if multilabel_train else get_to_data)(
                tup, vocab, labeldict, max_trlen))

    if multilabel_train:
        append_labelnums([x[-1] for x in stuffs['train']])

    if nonedenom > 0:
        # don't keep all the NONE labeled things
        trlabels = [x[-1] for x in stuffs['train']]
        none_idxs = [i for i, labellist in enumerate(trlabels) if
                     labellist[0] == labeldict["NONE"]]
        random.shuffle(none_idxs)
        # allow at most 1/(nonedenom+1) of NONE-labeled
        num_to_keep = int(
            math.floor((len(trlabels) - len(none_idxs)) / nonedenom))
        print("originally", len(trlabels), "training examples")
        print("keeping", num_to_keep, "NONE-labeled examples")
        ignore_idxs = set(none_idxs[num_to_keep:])

        # get rid of most of the NONE-labeled examples
        stuffs['train'] = [thing for i, thing in enumerate(stuffs['train']) if
                           i not in ignore_idxs]

    print(len(stuffs['train']), "training examples")

    if verbose:
        for _ in stuffs['train'][0]:
            print(_)

    for stage in ['valid', 'test']:
        # do val/test, which we also consider multilabel
        dataset = datasets[stage]
        max_len = max(len(tup[0]) for tup in dataset)
        for tup in dataset:
            stuffs[stage].extend(
                get_multilabeled_data(tup, vocab, labeldict, max_len))

        append_labelnums([x[-1] for x in stuffs[stage]])

        print(len(stuffs[stage]), "{} examples".format(stage))

    stage_to_abbr = {"train": "tr", "valid": "val", "test": "test"}
    h5fi = h5py.File(outfile, "w")
    for stage, stuff in stuffs.items():
        abbr = stage_to_abbr[stage]
        for name, content in zip(stuff_names, zip(*stuff)):
            h5fi["{}{}s".format(abbr, name)] = np.array(content, dtype=int)
    h5fi.close()

    # write dicts
    for d, name in ((vocab, 'dict'), (labeldict, 'labels')):
        revd = {v: k for k, v in d.items()}
        with open("{}.{}".format(outfile.split('.')[0], name), "w+") as f:
            for i in range(1, len(revd) + 1):
                f.write("%s %d \n" % (revd[i], i))


def convert_aux(additional):
    aux = 'NONE'
    if additional is None:
        aux = "NONE"
    elif isinstance(additional, str):
        aux = additional
    elif isinstance(additional, bool):
        if additional:
            aux = "HOME"
        else:
            aux = "AWAY"
    return aux


def filter_none_rels(rels):
    return filter(lambda rel: rel.type != "NONE", rels)


def data_to_csv_lines(data, filter_none=True):
    sent = ' '.join(data[0])
    rels = data[1]
    if filter_none:
        rels = filter_none_rels(rels)
    for rel in rels:
        additional = rel[3]
        aux = convert_aux(additional)
        yield '\t'.join(map(str, (sent, rel[0][0], rel[0][1], rel[0][2],
                                  rel[1][0], rel[1][1], rel[1][2], rel[2],
                                  aux)))


def make_translate_corpus(data, players, teams, cities, filter_none=True):
    tokens, rels = data
    res = []

    # add name information for copying
    for i, token in enumerate(tokens):
        if token in teams:
            rel_type = "TEAM_NAME"
        elif token in cities:
            rel_type = "TEAM_NAME"
        elif token in players:
            rel_type = "PLAYER_NAME"
        else:
            rel_type = None
        if rel_type is not None:
            res.append((i, rel_type, token, token))

    if filter_none:
        rels = filter_none_rels(rels)
    for rel in rels:
        ent = rel.ent.s
        num = str(rel.num.s)
        res.append((rel.num.start, rel.type, ent, num))

        # add home away information
        # additional = rel[3]
        # aux = convert_aux(additional)
        # if aux in ("HOME", "AWAY"):
        #     res.append((rel.num.start, "HOME_AWAY", ent, aux))

    res.sort()

    return res, tokens


# for extracting sentence-data pairs
def extract_sentence_data(outfile_prefix, path, connect_multiwords=True,
                          filter_max_len=50):
    datasets, all_ents, players, teams, cities = get_datasets(
        path, connect_multiwords=connect_multiwords, filter_none=True)
    for stage, dataset in datasets.items():
        # output translate data files
        corpus = map(
            lambda data: make_translate_corpus(data, players, teams, cities),
            dataset)
        corpus = filter(
            lambda pair: 0 < len(pair[0]) <= filter_max_len and len(
                pair[1]) <= filter_max_len,
            corpus)
        src_lines, tgt_lines = zip(*corpus)
        idxs, src_lines = zip(*map(
            lambda res: list(zip(*map(lambda t: (t[0], t[1:]), res))),
            src_lines))

        src_lines = map(
            lambda triples: ' '.join('|'.join((t[2], t[0], t[1]))
                                     for t in triples),
            src_lines)
        with open(outfile_prefix + '{}.src'.format(stage), 'w') as src_f:
            src_f.write('\n'.join(src_lines))

        tgt_lines = map(' '.join, tgt_lines)
        with open(outfile_prefix + '{}.tgt'.format(stage), 'w') as tgt_f:
            tgt_f.write('\n'.join(tgt_lines))

        sent_idx = list(zip(tgt_lines, idxs))
        with open(outfile_prefix + '{}.idx.json'.format(stage), 'w') as idx_f:
            json.dump(sent_idx, idx_f)

        # output csv
        with open(outfile_prefix + '{}.csv'.format(stage), 'w') as csv_f:
            for data in dataset:
                for line in data_to_csv_lines(data):
                    print(line, file=csv_f)


def prep_generated_data(genfile, dict_pfx, outfile, trdata, val_file,
                        rec_outfile=None, backup=False):
    # recreate vocab and labeldict
    def read_dict(s):
        d = {}
        with open("{}.{}".format(dict_pfx, s), "r") as f:
            for line in f:
                pieces = line.strip().split()
                d[pieces[0]] = int(pieces[1])
        return d

    vocab, labeldict = map(read_dict, ["dict", "labels"])

    with open(genfile, "r") as f:
        gens = f.readlines()

    all_ents, players, teams, cities = get_ents(trdata)

    if not backup:
        all_ents, players, teams, cities = (
            set(x.replace(' ', '_') for x in ents) for ents in
            (all_ents, players, teams, cities))
        extract = candidate_rels_extractor(all_ents, players, teams, cities,
                                           prons)

    with open(val_file, "r") as f:
        valdata = json.load(f) if backup else [
            [x.split('|') for x in line.split()] for line in f]

    assert len(valdata) == len(
        gens), "len(valdata) = {}, len(gens) = {}".format(len(valdata),
                                                          len(gens))

    # extract ent-num pairs from generated sentence
    nugz = []  # to hold (sentence_tokens, [rels]) tuples
    if not backup:
        gold_stats = []
        for entry, summ in zip(valdata, gens):
            gold_rels = []
            for rel in entry:
                if rel[1] not in ("TEAM_NAME", "PLAYER_NAME"):
                    gold_rels.append((rel[2], int(rel[0]), rel[1]))
            gold_matched = [0 for _ in gold_rels]

            sent = summ.split()
            ents = extract_entities(sent, all_ents, prons)
            nums = extract_numbers(sent)
            extracted_rels = []
            for ent in ents:
                for num in nums:
                    match = False
                    for rel_i, rel in enumerate(gold_rels):
                        if ent[2] == rel[0] and num[2] == rel[1]:
                            match = True
                            gold_matched[rel_i] = 1
                            extracted_rels.append(Rel(ent, num, rel[2], None))
                    if not match:
                        extracted_rels.append(Rel(ent, num, 'NONE', None))
            nugz.append((sent, extracted_rels))

            gold_stats.append((
                len(gold_rels), sum(gold_matched),
                sum(1 for rel in extracted_rels if rel[2] != 'NONE')))

        if rec_outfile is not None:
            line_format = '\t'.join
            print(line_format(('gold_n', 'gold_match_n', 'pred_nonnone_n')),
                  file=rec_outfile)
            for gold_stat in gold_stats:
                print(line_format(gold_stat), file=rec_outfile)
        ns = zip(*gold_stats)
        sums = tuple(map(sum, ns))
        ret = divide_or_const(sums[1], sums[0]), divide_or_const(sums[1],
                                                                 sums[2])
        print('gold recall: {:.6f}, cand prec: {:.6f}'.format(*ret))
    else:
        sent_reset_indices = {0}  # sentence indices where a box/story is reset
        for entry, summ in zip(valdata, gens):
            nugz.extend(extract(entry, summ, filter_none=False))
            sent_reset_indices.add(len(nugz))

    # save stuff
    max_len = max(len(tup[0]) for tup in nugz)
    p = []

    rel_reset_indices = []
    for t, tup in enumerate(nugz):
        # then last rel is the last of its box
        if not backup or t in sent_reset_indices:
            rel_reset_indices.append(len(p))
        p.extend(get_multilabeled_data(tup, vocab, labeldict, max_len))

    append_labelnums([x[-1] for x in p])

    print(len(p), "prediction examples")

    h5fi = h5py.File(outfile, "w")
    for name, content in zip(stuff_names, zip(*p)):
        h5fi["val{}s".format(name)] = np.array(content, dtype=int)
    h5fi["boxrestartidxs"] = np.array(np.array(rel_reset_indices),
                                      dtype=int)  # 1-indexed
    h5fi.close()

    if not backup:
        return ret


################################################################################

bs_keys = ["PLAYER-PLAYER_NAME", "PLAYER-START_POSITION", "PLAYER-MIN",
           "PLAYER-PTS",
           "PLAYER-FGM", "PLAYER-FGA", "PLAYER-FG_PCT", "PLAYER-FG3M",
           "PLAYER-FG3A",
           "PLAYER-FG3_PCT", "PLAYER-FTM", "PLAYER-FTA", "PLAYER-FT_PCT",
           "PLAYER-OREB",
           "PLAYER-DREB", "PLAYER-REB", "PLAYER-AST", "PLAYER-TO", "PLAYER-STL",
           "PLAYER-BLK",
           "PLAYER-PF", "PLAYER-FIRST_NAME", "PLAYER-SECOND_NAME"]

ls_keys = ["TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4",
           "TEAM-PTS", "TEAM-FG_PCT", "TEAM-FG3_PCT", "TEAM-FT_PCT", "TEAM-REB",
           "TEAM-AST", "TEAM-TOV", "TEAM-WINS", "TEAM-LOSSES", "TEAM-CITY",
           "TEAM-NAME"]

NUM_PLAYERS = 13


def get_player_idxs(entry):
    nplayers = 0
    home_players, vis_players = [], []
    for k, v in entry["box_score"]["PTS"].items():
        nplayers += 1

    num_home, num_vis = 0, 0
    for i in range(nplayers):
        player_city = entry["box_score"]["TEAM_CITY"][str(i)]
        if player_city == entry["home_city"]:
            if len(home_players) < NUM_PLAYERS:
                home_players.append(str(i))
                num_home += 1
        else:
            if len(vis_players) < NUM_PLAYERS:
                vis_players.append(str(i))
                num_vis += 1
    return home_players, vis_players


def box_preproc2(trdata):
    """
    just gets src for now
    """
    srcs = [[] for i in range(2 * NUM_PLAYERS + 2)]

    for entry in trdata:
        home_players, vis_players = get_player_idxs(entry)
        for ii, player_list in enumerate([home_players, vis_players]):
            for j in range(NUM_PLAYERS):
                src_j = []
                player_key = player_list[j] if j < len(player_list) else None
                for k, key in enumerate(bs_keys):
                    rulkey = key.split('-')[1]
                    val = entry["box_score"][rulkey][
                        player_key] if player_key is not None else "N/A"
                    src_j.append(val)
                srcs[ii * NUM_PLAYERS + j].append(src_j)

        home_src, vis_src = [], []
        for k in range(len(bs_keys) - len(ls_keys)):
            home_src.append("PAD")
            vis_src.append("PAD")

        for k, key in enumerate(ls_keys):
            home_src.append(entry["home_line"][key])
            vis_src.append(entry["vis_line"][key])

        srcs[-2].append(home_src)
        srcs[-1].append(vis_src)

    return srcs


def linearized_preproc(srcs):
    """
    maps from a num-rows length list of lists of ntrain to an
    ntrain-length list of concatenated rows
    """
    lsrcs = []
    for i in range(len(srcs[0])):
        src_i = []
        for j in range(len(srcs)):
            src_i.extend(srcs[j][i][1:])  # b/c in lua we ignore first thing
        lsrcs.append(src_i)
    return lsrcs


def fix_target_idx(summ, assumed_idx, word, neighborhood=5):
    """
    Tokenization can mess stuff up, so look around
    """
    for i in range(1, neighborhood + 1):
        if assumed_idx + i < len(summ) and summ[assumed_idx + i] == word:
            return assumed_idx + i
        elif 0 <= assumed_idx - i < len(summ) and summ[assumed_idx - i] == word:
            return assumed_idx - i
    return None


# for each target word want to know where it could've been copied from
def make_pointerfi(outfi, trdata, resolve_prons=False):
    """
    N.B. this function only looks at string equality in determining pointerness.
    this means that if we sneak in pronoun strings as their referents, we won't
    point to the pronoun if the referent appears in the table;
    we may use this tho to point to the correct number
    """
    rulsrcs = linearized_preproc(box_preproc2(trdata))

    all_ents, players, teams, cities = get_ents(trdata)

    skipped = 0

    train_links = []
    for i, entry in enumerate(trdata):
        home_players, vis_players = get_player_idxs(entry)
        inv_home_players = {pkey: jj for jj, pkey in enumerate(home_players)}
        inv_vis_players = {pkey: (jj + NUM_PLAYERS) for jj, pkey in
                           enumerate(vis_players)}
        summ = " ".join(entry['summary'])
        sents = sent_tokenize(summ)
        words_so_far = 0
        links = []
        prev_ents = []
        for j, sent in enumerate(sents):
            tokes = word_tokenize(
                sent)  # just assuming this gives me back original tokenization
            ents = extract_entities(tokes, all_ents, prons, prev_ents,
                                    resolve_prons,
                                    players, teams, cities)
            if resolve_prons:
                prev_ents.append(ents)
            nums = extract_numbers(tokes)
            # should return a list of (enttup, numtup, rel-name, identifier)
            # for each rel licensed by the table
            rels = get_rels(entry, tokes, ents, nums, players, teams, cities)
            for (enttup, numtup, label, idthing) in rels:
                if label != 'NONE':
                    # try to find corresponding words (for both ents and nums)
                    ent_start, ent_end, entspan, _ = enttup
                    num_start, num_end, numspan = numtup
                    if isinstance(idthing, bool):  # city or team
                        # get entity indices if any
                        for k, word in enumerate(tokes[ent_start:ent_end]):
                            src_idx = None
                            if word == entry["home_name"]:
                                src_idx = (2 * NUM_PLAYERS + 1) * (
                                        len(bs_keys) - 1) - 1  # last thing
                            elif word == entry["home_city"]:
                                src_idx = (2 * NUM_PLAYERS + 1) * (len(
                                    bs_keys) - 1) - 2  # second to last thing
                            elif word == entry["vis_name"]:
                                src_idx = (2 * NUM_PLAYERS + 2) * (
                                        len(bs_keys) - 1) - 1  # last thing
                            elif word == entry["vis_city"]:
                                src_idx = (2 * NUM_PLAYERS + 2) * (len(
                                    bs_keys) - 1) - 2  # second to last thing
                            if src_idx is not None:
                                targ_idx = words_so_far + ent_start + k
                                if targ_idx >= len(entry["summary"]) or \
                                        entry["summary"][targ_idx] != word:
                                    targ_idx = fix_target_idx(entry["summary"],
                                                              targ_idx, word)
                                if targ_idx is None:
                                    skipped += 1
                                else:
                                    assert rulsrcs[i][src_idx] == word and \
                                           entry["summary"][targ_idx] == word
                                    links.append((src_idx, targ_idx))

                                    # get num indices if any
                        for k, word in enumerate(tokes[num_start:num_end]):
                            src_idx = None
                            if idthing:  # home, so look in the home row
                                if entry["home_line"][label] == word:
                                    col_idx = ls_keys.index(label)
                                    src_idx = 2 * NUM_PLAYERS * (
                                            len(bs_keys) - 1) + len(
                                        bs_keys) - len(
                                        ls_keys) + col_idx - 1
                            else:
                                if entry["vis_line"][label] == word:
                                    col_idx = ls_keys.index(label)
                                    src_idx = (2 * NUM_PLAYERS + 1) * (
                                            len(bs_keys) - 1) + len(
                                        bs_keys) - len(
                                        ls_keys) + col_idx - 1
                            if src_idx is not None:
                                targ_idx = words_so_far + num_start + k
                                if targ_idx >= len(entry["summary"]) or \
                                        entry["summary"][targ_idx] != word:
                                    targ_idx = fix_target_idx(entry["summary"],
                                                              targ_idx, word)

                                if targ_idx is None:
                                    skipped += 1
                                else:
                                    assert rulsrcs[i][src_idx] == word and \
                                           entry["summary"][targ_idx] == word
                                    links.append((src_idx, targ_idx))
                    else:  # players
                        # get row corresponding to this player
                        player_row = None
                        if idthing in inv_home_players:
                            player_row = inv_home_players[idthing]
                        elif idthing in inv_vis_players:
                            player_row = inv_vis_players[idthing]
                        if player_row is not None:
                            # ent links
                            for k, word in enumerate(tokes[ent_start:ent_end]):
                                src_idx = None
                                if word == entry["box_score"]["FIRST_NAME"][
                                    idthing]:
                                    # second to last thing
                                    src_idx = (player_row + 1) * (len(
                                        bs_keys) - 1) - 2
                                elif word == entry["box_score"]["SECOND_NAME"][
                                    idthing]:
                                    src_idx = (player_row + 1) * (len(
                                        bs_keys) - 1) - 1  # last thing
                                if src_idx is not None:
                                    targ_idx = words_so_far + ent_start + k
                                    if entry["summary"][targ_idx] != word:
                                        targ_idx = fix_target_idx(
                                            entry["summary"], targ_idx, word)
                                    if targ_idx is None:
                                        skipped += 1
                                    else:
                                        assert rulsrcs[i][src_idx] == word and \
                                               entry["summary"][
                                                   targ_idx] == word
                                        # src_idx, target_idx
                                        links.append((src_idx,
                                                      targ_idx))
                            # num links
                            for k, word in enumerate(tokes[num_start:num_end]):
                                src_idx = None
                                if word == \
                                        entry["box_score"][label.split('-')[1]][
                                            idthing]:
                                    # subtract 1 because we ignore first col
                                    src_idx = player_row * (len(
                                        bs_keys) - 1) + bs_keys.index(
                                        label) - 1
                                if src_idx is not None:
                                    targ_idx = words_so_far + num_start + k
                                    if targ_idx >= len(entry["summary"]) or \
                                            entry["summary"][targ_idx] != word:
                                        targ_idx = fix_target_idx(
                                            entry["summary"], targ_idx, word)

                                    if targ_idx is None:
                                        skipped += 1
                                    else:
                                        assert rulsrcs[i][src_idx] == word and \
                                               entry["summary"][
                                                   targ_idx] == word
                                        links.append((src_idx, targ_idx))

            words_so_far += len(tokes)
        train_links.append(links)
    print("SKIPPED", skipped)

    # collapse multiple links
    trlink_dicts = []
    for links in train_links:
        links_dict = defaultdict(list)
        [links_dict[targ_idx].append(src_idx) for src_idx, targ_idx in links]
        trlink_dicts.append(links_dict)

    # write in fmt:
    # targ_idx,src_idx1[,src_idx...]
    with open(outfi, "w+") as f:
        for links_dict in trlink_dicts:
            targ_idxs = sorted(links_dict.keys())
            fmtd = [",".join([str(targ_idx)] + [str(thing) for thing in
                                                set(links_dict[targ_idx])])
                    for targ_idx in targ_idxs]
            f.write("%s\n" % " ".join(fmtd))


# for coref prediction stuff
# we'll use string equality for now
def save_coref_task_data(outfile, in_file="full_newnba_prepdata2.json"):
    with open(in_file, "r") as f:
        data = json.load(f)

    all_ents, players, teams, cities = get_ents(data["train"])
    datasets = []

    # labels are nomatch, match, pron
    for dataset in [data["train"], data["valid"]]:
        examples = []
        for i, entry in enumerate(dataset):
            summ = entry["summary"]
            ents = extract_entities(summ, all_ents, prons)
            for j in range(1, len(ents)):
                # just get all the words from previous mention till this
                # one starts
                prev_start, prev_end, prev_str, _ = ents[j - 1]
                curr_start, curr_end, curr_str, curr_pron = ents[j]
                # window = summ[prev_start:curr_start]
                window = summ[prev_end:curr_start]
                label = None
                if curr_pron:  # prons
                    label = 3
                else:
                    # label = 2 if prev_str == curr_str else 1
                    label = 2 if prev_str in curr_str or curr_str \
                                 in prev_str else 1
                examples.append((window, label))
        datasets.append(examples)

    # make vocab and get labels
    word_counter = Counter()
    [word_counter.update(tup[0]) for tup in datasets[0]]
    for k in word_counter.keys():
        if word_counter[k] < 2:
            del word_counter[k]  # will replace w/ unk
    word_counter["UNK"] = 1
    vocab = dict(((wrd, i + 1) for i, wrd in enumerate(word_counter.keys())))
    labeldict = {"NOMATCH": 1, "MATCH": 2, "PRON": 3}

    max_trlen = max((len(tup[0]) for tup in datasets[0]))
    max_vallen = max((len(tup[0]) for tup in datasets[1]))
    print("max sentence lengths:", max_trlen, max_vallen)

    # map words to indices
    trwindows = [
        [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in window]
        + [-1] * (max_trlen - len(window)) for (window, label) in datasets[0]]
    trlabels = [label for (window, label) in datasets[0]]
    valwindows = [
        [vocab[wrd] if wrd in vocab else vocab["UNK"] for wrd in window]
        + [-1] * (max_vallen - len(window)) for (window, label) in datasets[1]]
    vallabels = [label for (window, label) in datasets[1]]

    print(len(trwindows), "training examples")
    print(len(valwindows), "validation examples")
    print(Counter(trlabels))
    print(Counter(vallabels))

    h5fi = h5py.File(outfile, "w")
    h5fi["trwindows"] = np.array(trwindows, dtype=int)
    h5fi["trlens"] = np.array([len(window) for (window, label) in datasets[0]],
                              dtype=int)
    h5fi["trlabels"] = np.array(trlabels, dtype=int)
    h5fi["valwindows"] = np.array(valwindows, dtype=int)
    h5fi["vallens"] = np.array([len(window) for (window, label) in datasets[1]],
                               dtype=int)
    h5fi["vallabels"] = np.array(vallabels, dtype=int)
    h5fi.close()

    # write dicts
    revvocab = dict(((v, k) for k, v in vocab.items()))
    revlabels = dict(((v, k) for k, v in labeldict.items()))
    with open(outfile.split('.')[0] + ".dict", "w+") as f:
        for i in range(1, len(revvocab) + 1):
            f.write("%s %d \n" % (revvocab[i], i))

    with open(outfile.split('.')[0] + ".labels", "w+") as f:
        for i in range(1, len(revlabels) + 1):
            f.write("%s %d \n" % (revlabels[i], i))


def mask_output(input_path, trdata):
    all_ents, players, teams, cities = get_ents(trdata)
    all_ents = {x.replace(' ', '_') for x in all_ents}

    with open(input_path, "r") as f:
        sents = f.readlines()

    masked_sents = []
    for idx, sent in enumerate(sents):
        sent = sent.split()
        ents = extract_entities(sent, all_ents, prons)
        nums = extract_numbers(sent)
        ranges = []
        for ent in ents:
            ranges.append((ent[0], ent[1], 'ENT'))
        for num in nums:
            ranges.append((num[0], num[1], 'NUM'))
        ranges.sort(key=lambda x: x[0])

        masked_sent = []
        i = 0
        while i < len(sent):
            match = False
            for r in ranges:
                if i == r[0]:
                    match = True
                    masked_sent.append(r[2])
                    i = r[1]
                    break
            if not match:
                masked_sent.append(sent[i])
                i += 1
        masked_sents.append(masked_sent)

    with open(input_path + '.masked', 'w') as f:
        f.write('\n'.join([' '.join(s) for s in masked_sents]))


def save_ent(output_path, trdata):
    all_ents, players, teams, cities = get_ents(trdata)
    all_ents = {x.replace(' ', '_') for x in all_ents}

    with open(output_path, 'w') as f:
        json.dump(list(all_ents), f)


def read_trdata():
    return get_json_dataset(args.input_path, 'train')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility Functions')
    parser.add_argument('mode', type=str, default='ptrs',
                        choices=['ptrs', 'make_ie_data', 'prep_gen_data',
                                 'extract_sent', 'mask', 'save_ent'],
                        help="what utility function to run")
    parser.add_argument('--input_path', type=str, default="rotowire",
                        help="path to input")
    parser.add_argument('--output', type=str, default="",
                        help="desired path to output file")
    parser.add_argument('--gen', type=str, default="",
                        help="path to file containing generated summaries")
    parser.add_argument('--dict_pfx', type=str, default="roto-ie",
                        help="prefix of .dict and .labels files")
    parser.add_argument('--val_file', type=str,
                        default=os.path.join("nba_data", "gold.valid.txt"),
                        help="file as reference in prep_gen_data mode, "
                             "of which every entry is in the form "
                             "entry|attribute|value")
    args = parser.parse_args()

    if args.mode == 'ptrs':
        make_pointerfi(args.output, read_trdata())
    elif args.mode == 'make_ie_data':
        save_full_sent_data(args.output, path=args.input_path,
                            multilabel_train=True)
    elif args.mode == 'prep_gen_data':
        prep_generated_data(args.gen, args.dict_pfx, args.output,
                            trdata=read_trdata(),
                            val_file=args.val_file)
    elif args.mode == 'extract_sent':
        extract_sentence_data(args.output, path=args.input_path)
    elif args.mode == 'mask':
        mask_output(args.input_path, read_trdata())
    elif args.mode == 'save_ent':
        save_ent(args.output, read_trdata())
