import math
import numpy as np
from utils import lcm
from collections import Counter


#Average happiness scores, more memory efficient
def ANH_SCORE(pred, gp, cp):
    gift_counts = Counter(elem[1] for elem in pred)
    for count in gift_counts.values():
        assert count <= 1000

    # check that they all have the same gifts
    for t1 in np.arange(0, 5001, 3):
        triplet1 = pred[t1]
        triplet2 = pred[t1 + 1]
        triplet3 = pred[t1 + 2]
        assert triplet1[1] == triplet2[1] and triplet2[1] == triplet3[1]

    for t1 in np.arange(5001, 45001, 2):
        twin1 = pred[t1]
        twin2 = pred[t1 + 1]
        assert twin1[1] == twin2[1]

    tch = 0
    tgh = np.zeros(1000)

    for row in pred:
        cid, gid = row

        assert cid < 1e6
        assert gid < 1000
        assert cid >= 0
        assert gid >= 0

        ch = (100 - np.where(gp[cid] == gid)[0]) * 2
        if not ch:
            ch = -1

        gh = (1000 - np.where(cp[gid] == cid)[0]) * 2
        if not gh:
            gh = -1

        tch += ch
        tgh[gid] += gh
    return float(math.pow(tch * 10, 3) + math.pow(np.sum(tgh), 3)) / 8e+27


# print(ANH_SCORE(test))

def ANH_SCORE_ROW(pred, gp, cp):
    tch = 0
    tgh = np.zeros(1000)
    for row in pred:
        cid, gid = row
        ch = (100 - np.where(gp[cid] == gid)[0]) * 2
        if not ch:
            ch = -1
        gh = (1000 - np.where(cp[gid] == cid)[0]) * 2
        if not gh:
            gh = -1
        tch += ch
        tgh[gid] += gh
    return float(math.pow(tch * 10, 3) + math.pow(np.sum(tgh),
                                                  3)) / 8e+27  # math.pow(float(tch)/2e8,2) + math.pow(np.mean(tgh)/2e6,2)

def metric_function(c1, c2,gp,cp):
    cid1, gid1 = c1
    cid2, gid2 = c2
    return [ANH_SCORE_ROW([c1,c2],gp,cp), ANH_SCORE_ROW([[cid1,gid2],[cid2,gid1]],gp,cp)]

def avg_normalized_happiness(pred, gift, wish):
    n_children = 1000000  # n children to give
    n_gift_type = 1000  # n types of gifts available
    n_gift_quantity = 1000  # each type of gifts are limited to this quantity
    n_gift_pref = 100  # number of gifts a child ranks
    n_child_pref = 1000  # number of children a gift ranks
    twins = math.ceil(0.04 * n_children / 2.) * 2  # 4% of all population, rounded to the closest number
    triplets = math.ceil(0.005 * n_children / 3.) * 3  # 0.5% of all population, rounded to the closest number
    ratio_gift_happiness = 2
    ratio_child_happiness = 2

    # check if triplets have the same gift
    for t1 in np.arange(0, triplets, 3):
        triplet1 = pred[t1]
        triplet2 = pred[t1 + 1]
        triplet3 = pred[t1 + 2]
        # print(t1, triplet1, triplet2, triplet3)
        assert triplet1 == triplet2 and triplet2 == triplet3

    # check if twins have the same gift
    for t1 in np.arange(triplets, triplets + twins, 2):
        twin1 = pred[t1]
        twin2 = pred[t1 + 1]
        # print(t1)
        assert twin1 == twin2

    max_child_happiness = n_gift_pref * ratio_child_happiness
    max_gift_happiness = n_child_pref * ratio_gift_happiness
    total_child_happiness = 0
    total_gift_happiness = np.zeros(n_gift_type)

    for i in range(len(pred)):
        child_id = i
        gift_id = pred[i]

        # check if child_id and gift_id exist
        assert child_id < n_children
        assert gift_id < n_gift_type
        assert child_id >= 0
        assert gift_id >= 0
        child_happiness = (n_gift_pref - np.where(wish[child_id] == gift_id)[0]) * ratio_child_happiness
        if not child_happiness:
            child_happiness = -1

        gift_happiness = (n_child_pref - np.where(gift[gift_id] == child_id)[0]) * ratio_gift_happiness
        if not gift_happiness:
            gift_happiness = -1

        total_child_happiness += child_happiness
        total_gift_happiness[gift_id] += gift_happiness

    denominator1 = n_children * max_child_happiness
    denominator2 = n_gift_quantity * max_gift_happiness * n_gift_type
    common_denom = lcm(denominator1, denominator2)
    multiplier = common_denom / denominator1

    print(multiplier, common_denom)
    child_hapiness = math.pow(total_child_happiness * multiplier, 3) / float(math.pow(common_denom, 3))
    santa_hapiness = math.pow(np.sum(total_gift_happiness), 3) / float(math.pow(common_denom, 3))
    print('Child hapiness: {}'.format(child_hapiness))
    print('Santa hapiness: {}'.format(santa_hapiness))
    ret = child_hapiness + santa_hapiness
    return ret



#optimized this
def get_overall_hapiness(wish, gift,triplet_stop=5001,twin_stop=45001):
    #input, list of all wishes
    #list of all gifts

    list_limit = wish.shape[1]
    #list_limit = 42

    #convert to dict comprehension
    res_child = dict()
    for i in range(0, triplet_stop):
        app = i - (i % 3)
        for j in range(list_limit):
            if (app, wish[i][j]) in res_child:
                res_child[(app, wish[i][j])] += 10 * (1 + (wish.shape[1] - j) * 2)
            else:
                res_child[(app, wish[i][j])]  = 10 * (1 + (wish.shape[1] - j) * 2)

    for i in range(triplet_stop, twin_stop):
        app = i + (i % 2)
        for j in range(list_limit):
            if (app, wish[i][j]) in res_child:
                res_child[(app, wish[i][j])] += 10 * (1 + (wish.shape[1] - j) * 2)
            else:
                res_child[(app, wish[i][j])]  = 10 * (1 + (wish.shape[1] - j) * 2)

    for i in range(twin_stop, wish.shape[0]):
        app = i
        for j in range(list_limit):
            res_child[(app, wish[i][j])]  = 10 * (1 + (wish.shape[1] - j) * 2)

    res_santa = dict()
    for i in range(gift.shape[0]):
        for j in range(gift.shape[1]):
            cur_child = gift[i][j]
            if cur_child < triplet_stop:
                cur_child -= cur_child % 3
            elif cur_child < twin_stop:
                cur_child += cur_child % 2
            res_santa[(cur_child, i)] = (1 + (gift.shape[1] - j)*2)

    positive_cases = list(set(res_santa.keys()) | set(res_child.keys()))
    print('Positive case tuples (child, gift): {}'.format(len(positive_cases)))

    res = dict()
    for p in positive_cases:
        res[p] = 0
        if p in res_child:
            a = res_child[p]
            res[p] += int((a ** 3) * 4)
        if p in res_santa:
            b = res_santa[p]
            res[p] += int((b ** 3) / 4)

    return res