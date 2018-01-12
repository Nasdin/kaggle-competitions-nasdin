import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from happiness import avg_normalized_happiness



#competition parameters
n_children = 1000000
n_gift_type = 1000
n_gift_quantity = 1000
n_child_wish = 100
triplets = 5001
twins = 40000
tts = triplets + twins


child_data = pd.read_csv('data/child_wishlist_v2.csv', 
                         header=None).drop(0, 1).values
gift_data = pd.read_csv('data/gift_goodkids_v2.csv', 
                        header=None).drop(0, 1).values


def optimize_block(child_block, current_gift_ids):
    gift_block = current_gift_ids[child_block]
    C = np.zeros((block_size, block_size))
    for i in range(block_size):
        c = child_block[i]
        for j in range(block_size):
            g = gift_ids[gift_block[j]]
            C[i, j] = child_happiness[c][g]
    row_ind, col_ind = linear_sum_assignment(C)
    return (child_block[row_ind], gift_block[col_ind])



#Building an array of initial child and gift happiness
gift_happiness = (1. / (2 * n_gift_type)) * np.ones(
    shape=(n_gift_type, n_children), dtype = np.float32)

for g in range(n_gift_type):
    for i, c in enumerate(gift_data[g]):
        gift_happiness[g,c] = -2. * (n_gift_type - i)  

child_happiness = (1. / (2 * n_child_wish)) * np.ones(
    shape=(n_children, n_gift_type), dtype = np.float32)

for c in range(n_children):
    for i, g in enumerate(child_data[c]):
        child_happiness[c,g] = -2. * (n_child_wish - i) 

gift_ids = np.array([[g] * n_gift_quantity for g in range(n_gift_type)]).flatten()





initial_sub = 'nas.csv'
subm = pd.read_csv(initial_sub)
subm['gift_rank'] = subm.groupby('GiftId').rank() - 1
subm['gift_id'] = subm['GiftId'] * 1000 + subm['gift_rank']
subm['gift_id'] = subm['gift_id'].astype(np.int64)
current_gift_ids = subm['gift_id'].values



wish = pd.read_csv('data/child_wishlist_v2.csv', 
                   header=None).as_matrix()[:, 1:]
gift_init = pd.read_csv('data/gift_goodkids_v2.csv', 
                        header=None).as_matrix()[:, 1:]
gift = gift_init.copy()
answ_org = np.zeros(len(wish), dtype=np.int32)
answ_org[subm[['ChildId']]] = subm[['GiftId']]
score_org = avg_normalized_happiness(answ_org, gift, wish)

print('Predicted score: {:.8f}'.format(score_org))





  

block_size = 1500
n_blocks = int((n_children - tts) / block_size)
children_rmd = 1000000 - 45001 - n_blocks * block_size
print('block size: {}, num blocks: {}, children reminder: {}'.
      format(block_size, n_blocks, children_rmd))




answ_iter = np.zeros(len(wish), dtype=np.int32)
score_best = score_org
subm_best = subm
perm_len = 100
block_len = 5
for i in range(perm_len):  
    print('Current permutation step is: %d' %(i+1))
    child_blocks = np.split(np.random.permutation
                            (range(tts, n_children - children_rmd)), n_blocks)
    for child_block in tqdm(child_blocks[:block_len]):
        start_time = dt.datetime.now()
        cids, gids = optimize_block(child_block, current_gift_ids=current_gift_ids)
        current_gift_ids[cids] = gids
        end_time = dt.datetime.now()
        print('Time spent to optimize this block in seconds: {:.2f}'.
              format((end_time-start_time).total_seconds()))
        ## need evaluation step for every block iteration 
        subm['GiftId'] = gift_ids[current_gift_ids]
        answ_iter[subm[['ChildId']]] = subm[['GiftId']]
        score_iter = avg_normalized_happiness(answ_iter, gift, wish)
        print('Score achieved in current iteration: {:.8f}'.format(score_iter))
        if score_iter > score_best:
            subm_best['GiftId'] = gift_ids[current_gift_ids]
            score_best = score_iter
            print('This is a performance improvement!')
            subm_best[['ChildId', 'GiftId']].to_csv('improved_sub.csv', index=False)
        else: print('No improvement at this iteration!')
            
subm_best[['ChildId', 'GiftId']].to_csv('improved_sub.csv', index=False)
print('Best score achieved is: {:.8f}'.format(score_best))


