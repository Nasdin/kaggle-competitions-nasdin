from sklearn.utils import shuffle
from collections import Counter
from random import randint
import pandas as pd
import numpy as np
import copy

gp = pd.read_csv('../input/santa-gift-matching/child_wishlist.csv',header=None).drop(0, 1).values
cp = pd.read_csv('../input/santa-gift-matching/gift_goodkids.csv',header=None).drop(0, 1).values
test = pd.read_csv('../input/uniform-filling-of-gift-buckets/uniform_fill.csv').values.tolist()

def ANH_SCORE(pred):
    gift_counts = Counter(elem[1] for elem in pred)
    for count in gift_counts.values():
        assert count <= 1000

    for t1 in range(0,4000,2):
        twin1 = pred[t1]
        twin2 = pred[t1+1]
        assert twin1[1] == twin2[1]
    
    tch = 0
    tgh = np.zeros(1000)
    
    for row in pred:
        cid, gid = row

        assert cid < 1e6
        assert gid < 1000
        assert cid >= 0 
        assert gid >= 0
        
        ch = (10 - np.where(gp[cid]==gid)[0]) * 2
        if not ch:
            ch = -1

        gh = (1000 - np.where(cp[gid]==cid)[0]) * 2
        if not gh:
            gh = -1

        tch += ch
        tgh[gid] += gh
    return float(tch)/2e7 + np.mean(tgh) / 2e6

#print(ANH_SCORE(test, cp, gp))

def ANH_SCORE_ROW(pred):
    tch = 0
    tgh = np.zeros(1000)
    for row in pred:
        cid, gid = row
        ch = (10 - np.where(gp[cid]==gid)[0]) * 2
        if not ch:
            ch = -1
        gh = (1000 - np.where(cp[gid]==cid)[0]) * 2
        if not gh:
            gh = -1
        tch += ch
        tgh[gid] += gh
    return float(tch)/2e7 + np.mean(tgh) / 2e6

def metric_function(c1, c2):
    cid1, gid1 = c1
    cid2, gid2 = c2
    return [ANH_SCORE_ROW([c1,c2]), ANH_SCORE_ROW([[cid1,gid2],[cid2,gid1]])]

def objective_function_swap(test):
    for b in range(4000,len(test),444): #skip twins for now 4000
        for j in range(900000,len(test),333): #start at last iteration
            mf = metric_function(test[b], test[j])
            if mf[0] < mf[1]:
                temp = int(test[b][1])
                test[b][1] = int(test[j][1])
                test[j][1] = temp
                print(b, j, 'ANH', ANH_SCORE(test))
                break
        if b % 100000 == 0:
            print(b, j, 'ANH', ANH_SCORE(test))
    return test

for i in range(1):
    print(i, '='*20)
    objective_function_swap(test)

    print('eval')
    score = ANH_SCORE(test)
    print('Predicted score: {:.8f}'.format(score))
    
    out = open('01_public_subm.csv', 'w')
    out.write('ChildId,GiftId\n')
    for i in range(len(test)):
        out.write(str(test[i][0]) + ',' + str(test[i][1]) + '\n')
    out.close()