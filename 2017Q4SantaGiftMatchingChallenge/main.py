import numpy as np
import pandas as pd
import gc
from happiness import get_overall_hapiness
from happiness import avg_normalized_happiness
from ortools.graph import pywrapgraph


data_path = 'data/'
wishlist_file = 'child_wishlist_v2.csv'
goodkids_file = 'gift_goodkids_v2.csv'

#Competition Parameters
n_children = 1000000 #1 million
n_gift_type = 1000
n_gift_quantity = 1000
n_child_wish = 100
triplets_percentage = 0.5 #0.5%
twins_percentage = 4 #4%

#induction
triplets_ = int((triplets_percentage/100 ) * n_children) #amount of triplets
twins_ = int((twins_percentage/100 ) * n_children) -1 #amount of twins (no. 5001 to 45000) so 4% -1

#the start points
triplets = 0 #starts at 0
twins = triplets + triplets_ +1 #the start point of twins, ending of triplets
tts = twins_ + twins  #the start point of children without clones ( but its 5001 - 45000)



def solve():
    #the index is the label
    wish = pd.read_csv(data_path+wishlist_file, header=None).as_matrix()[:, 1:]
    gift = pd.read_csv(data_path+goodkids_file, header=None).as_matrix()[:, 1:]
    #default the answers
    answ = np.empty(len(wish), dtype=np.int32)
    answ[:] = -1
    #calculate the initial happiness
    happiness = get_overall_hapiness(wish, gift)
    gc.collect() #garbage collector

    # Give initial values
    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []
    supplies = []

    #traditional design pattern
    min_h = 10**100
    max_h = -10**100
    avg_h = 0

    #converted to list comprehension

    for h in happiness:
        c, g = h

        start_nodes.append(int(c))
        end_nodes.append(int(n_children + g))
        if c < twins:
            capacities.append(3)
        elif c < tts:
            capacities.append(2)
        else:
            capacities.append(1)
        unit_costs.append(-happiness[h])
        if happiness[h] > max_h:
            max_h = happiness[h]
        if happiness[h] < min_h:
            min_h = happiness[h]
        avg_h += happiness[h]
    print('Max single happiness: {}'.format(max_h))
    print('Min single happiness: {}'.format(min_h))
    print('Avg single happiness: {}'.format(avg_h / len(happiness)))

    for i in range(n_children):
        if i < twins:
            supplies.append(3)
        elif i < tts:
            supplies.append(2)
        else:
            supplies.append(1)
    for j in range(n_children, 1001000):
        supplies.append(-1000)

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # Add each arc.
    for i in range(0, len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i], capacities[i], unit_costs[i])

    # Add node supplies.
    for i in range(0, len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    # Find the minimum cost flow
    print('Start solve....')
    min_cost_flow.SolveMaxFlowWithMinCost()
    res1 = min_cost_flow.MaximumFlow()
    print('Maximum flow:', res1)
    res2 = min_cost_flow.OptimalCost()
    print('Optimal cost:', -res2 / 2000000000)
    print('Num arcs:', min_cost_flow.NumArcs())

    total = 0
    for i in range(min_cost_flow.NumArcs()):
        cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
        if cost != 0:
            answ[min_cost_flow.Tail(i)] = min_cost_flow.Head(i) - n_children
            total += 1
    print('Assigned: {}'.format(total))

    print('Check for overflow...')
    gift_count = np.zeros(1000, dtype=np.int32)
    for i in range(len(answ)):
        if answ[i] != -1:
            gift_count[answ[i]] += 1
    for i in range(1000):
        if gift_count[i] > 1000:
            print('Gift error: {} (Value: {})'.format(i, gift_count[i]))


    #this code basically just takes the gift of the first twin/triplet..
    #A better way would be to consider the overall happiness of all three.
    # Add triplets restrictions
    for i in range(0, twins, 3):
        answ[i + 1] = answ[i]
        answ[i + 2] = answ[i]

    # Add twins restrictions
    for i in range(twins, tts, 2):
        answ[i] = answ[i + 1]

    if answ.min() == -1:
        print('Some children without present')
        exit()

    print('Check for overflow after twins/triplets assigned')
    gift_count = np.zeros(1000, dtype=np.int32)
    for i in range(len(answ)):
        gift_count[answ[i]] += 1

    ov_count = 0
    for i in range(1000):
        if gift_count[i] > 1000:
            # print('Gift error: {} (Value: {})'.format(i, gift_count[i]))
            ov_count += 1
    if gift_count.max() > 1000:
        print('Gift overflow! Count: {}'.format(ov_count))

    for i in range(tts, len(answ)):
        if gift_count[answ[i]] > 1000:
            old_val = answ[i]
            j = np.argmin(gift_count)
            answ[i] = j
            gift_count[old_val] -= 1
            gift_count[j] += 1

    print('Check for overflow after simple fix')
    gift_count = np.zeros(1000, dtype=np.int32)
    for i in range(len(answ)):
        gift_count[answ[i]] += 1

    ov_count = 0
    for i in range(1000):
        if gift_count[i] > 1000:
            print('Gift error: {} (Value: {})'.format(i, gift_count[i]))
            ov_count += 1
    if gift_count.max() > 1000:
        print('Gift overflow! Count: {}'.format(ov_count))
        exit()

    print('Start score calculation...')
    score = avg_normalized_happiness(answ, gift, wish)
    print('Predicted score: {:.12f}'.format(score))

    out = open('subm_{:.12f}.csv'.format(score), 'w')
    out.write('ChildId,GiftId\n')
    for i in range(len(answ)):
        out.write(str(i) + ',' + str(answ[i]) + '\n')
    out.close()



if __name__ == '__main__':
    solve()