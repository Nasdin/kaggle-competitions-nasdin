from sklearn.utils import shuffle
from multiprocessing import *
import pandas as pd
import numpy as np
from happiness import metric_function, ANH_SCORE

#What this does
#Takes an existing submission and reworks the submission and sees if it can improve the score

#constant values
randomness = 2019
gp = pd.read_csv('data/child_wishlist_v2.csv', header=None).drop(0, 1).values
cp = pd.read_csv('data/gift_goodkids_v2.csv', header=None).drop(0, 1).values


def objective_function_swap(otest,gp=gp,cp=cp):
    otest = otest.values
    otest = shuffle(otest, random_state=randomness)
    #score1 = ANH_SCORE_ROW(otest)
    for b in range(len(otest)):
        for j in range(b+1,len(otest)):
            mf = metric_function(otest[b], otest[j],gp,cp)
            if mf[0] < mf[1]:
                temp = int(otest[b][1])
                otest[b][1] = int(otest[j][1])
                otest[j][1] = temp
                break
    #score2 = ANH_SCORE_ROW(otest)
    #if score2 > score1:
        #print(score2 - score1)
    otest = pd.DataFrame(otest)
    return otest

def multi_transform(mtest):
    p = Pool(cpu_count())
    mtest = p.map(objective_function_swap, np.array_split(mtest, cpu_count()*30))
    mtest = pd.concat(mtest, axis=0, ignore_index=True).reset_index(drop=True)
    p.close(); p.join()
    return mtest


#If this was run standalone, it mean we're taking an existing submission and improving upon it.
if __name__ == '__main__':
    #Submission to be improved on
    test = pd.read_csv('nas.csv')
    #get initial score
    score = ANH_SCORE(test.values,gp,cp)



    #optimize the non clone children

    #loop it to optimize for this amount of times
    for i in range(10000):
        test2 = multi_transform(shuffle(test[45001:100000].copy(), random_state=randomness))
        test3 = pd.concat([pd.DataFrame(test[:45001].values), pd.DataFrame(test2), pd.DataFrame(test[100000:].values)], axis=0, ignore_index=True).reset_index(drop=True).values
        test3 = pd.DataFrame(test3)
        test3.columns = ['ChildId','GiftId']
        #checking the score if it improved
        score_ = ANH_SCORE(test3.values,gp,cp)

        #if improves, this becomes the new score
        if score_ > score:
            #overwrite it
            test=test3
            score = score_
            test.to_csv('nas.csv', index=False)
            print("Score Improved, new score =",score_,"by",(score_-score)/score*100,"%")



        #save the progress so we can stop at anytime.
        randomness+=1 #try a different seed
        print("finished",i+1,'loops')

    print("finished optimizing, getting score")
    print(ANH_SCORE(test.values,gp,cp))

