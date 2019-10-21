import numpy as np
import pandas as pd
import argparse
import time


# Helper function to read file : 'GroceryStoreDataSet.csv' from Kaggle
def k_DataReader(filename):
    '''
    Parameter :
            filename(str): full file name
    Return:
            python dictionary of sets { Transaction ID : set( Item ID ) }
    '''
    transactions = dict()
    with open(filename, 'r') as f:
        for count, line in enumerate(f, 1):
            # count is Transaction ID and key in dict
            line = line.strip('"\n').split(',')
            transactions.update({count : set(line)})

    return transactions

def IBM_DataReader( filename ):
    '''
    Helper function to read IBM Quest Synthetic data.
    It receive a filename and return a dictionary 
    { Transaction ID : set( Item ID ) }
    Parameter : 
        filename (str) : data filename (ex: 'input.data', 'input.csv')
    Return value:
        a dictionary 
    '''
    transactions = dict()
    with open(filename, 'r') as f:
        for line in f:
            obj = line.split()
            TID = obj[1]
            ItemID = obj[2]
            if TID in transactions: # check key TID exist or not
                transactions[TID].update({ItemID})
            else: # key value pair dosen't exist then update dictionary 
                transactions.update({TID : {ItemID} })    
    return transactions

def rules_generation(a, support, min_confidence = 0.2):
    rules = pd.DataFrame({'X':[], 'Y':[], 'confidence':[], 'support':[]})
    for p in range(len(a)):
        for m in range(len(a)):
            if p == m: continue
            temp = set(a[p]) | set(a[m])
            if len(temp) == len(a[p]) + len(a[m]) and temp in a and (support[a.index(temp)] / support[p]) >= min_confidence:
                rules = rules.append({
                    'X': a[p],
                    'Y': a[m],
                    'confidence': support[a.index(temp)] / support[p],
                    'support': support[a.index(temp)]
                }, ignore_index=True)

    return rules

def apriori(transactions, min_support = 0.2):
    min_count = len(transactions) * min_support

    # obtain every unique item from the transactions
    def unique_item(itemset):
        uni_item = []
        for i in itemset:
            uni_item.extend(itemset[i])
        return list(set(uni_item))
    # get the initial item set
    items = unique_item(transactions)

    def gen_Candidate(L):
        candidate = []
        for l in largeitemset:
            for item in items:
                temp = list(set(l) | {item})
                if temp in candidate: continue
                in_L = True
                for diff in l:
                    check = list(set(temp) - {diff})
                    if check not in largeitemset: in_L = False
                if in_L: candidate.append(set(temp))
        return candidate
    
    def gen_Largeitemset(candidate):
        count = np.zeros(len(candidate))
        for i in range(len(candidate)):
            for j in transactions:
                if set(candidate[i]).issubset(set(transactions[j])): count[i] += 1
        return np.array(candidate)[count >= min_count].tolist(), count[count >= min_count].tolist()

    candidate = []
    for i in items: candidate.append([i])
    largeitemset, support_count = gen_Largeitemset(candidate)
    ans = largeitemset

    while len(largeitemset) != 0:
        candidate = gen_Candidate(largeitemset)
        largeitemset, support = gen_Largeitemset(candidate)
        ans = ans + largeitemset
        support_count = support_count + support
    return ans, support_count


if __name__ == '__main__':
    tStart = time.time()
    parser = argparse.ArgumentParser(description='dean')
    parser.add_argument('dataset', choices=['kaggle', 'IBM'])
    parser.add_argument("--rules", default=False, action='store_true')
    args = parser.parse_args()

    if args.dataset == 'kaggle':
        filename = 'data/GroceryStoreDataSet.csv'
        DB = k_DataReader(filename)
        DB_new = []
        for key, item in DB.items():
            #print(item)
            DB_new.append(item)

    if args.dataset == 'IBM':
        filename = 'data/output.data'
        DB = IBM_DataReader(filename)
        DB_new = []
        for key, item in DB.items():
            #print(item)
            DB_new.append(item)


    print(DB)
    print('Apriori\n===')
    a, support = apriori(DB, 0.2)
    for i in range(len(a)):
        print('set :', a[i], 'support :', support[i])

    if args.rules:
        print('\nRules Generation(X -> Y)\n===')
        rules = rules_generation(a, support)
        print(rules.sort_values(by=['confidence'], ascending=False))

    tEnd = time.time()
    print ("This code cost %f sec." % (tEnd - tStart))
