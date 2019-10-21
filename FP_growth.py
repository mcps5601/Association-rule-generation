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

#variables:
#name of the node, a count
#nodelink used to link similar items
#parent vaiable used to refer to the parent of the node in the tree
#node contains an empty dictionary for the children in the node
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode      #needs to be updated
        self.children = {}

    #increments the count variable with a given amount
    def inc(self, numOccur):
        self.count += numOccur

    #Helper function for displaying tree in text. Useful for debugging.
    def disp(self, ind=1):
        print ('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)
def createTree(dataSet, minSup=1): #create FP-tree from dataset but don't mine
    headerTable = {}
    
    #go over dataSet twice
    for trans in dataSet: #first pass counts frequency of occurance
        for item in trans:
            # Return 0 if item not in the headerTable
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
            
    for k in list(headerTable):  #remove items not meeting minSup
        if headerTable[k] < minSup: 
            del(headerTable[k])
            
    freqItemSet = set(headerTable.keys())
    
    if len(freqItemSet) == 0: return None, None  #if no items meet min support -->get out
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] #reformat headerTable to use Node link 
        
    #print 'headerTable: ',headerTable
    retTree = treeNode('Null Set', 1, None) #create tree
    for tranSet, count in dataSet.items():  #go through dataset 2nd time
        localD = {}
        for item in tranSet:  #put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)#populate tree with ordered freq itemset
    return retTree, headerTable #return tree and header table


def updateTree(items, inTree, headerTable, count):

    if items[0] in inTree.children:#check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count) #incrament count

    else:   #add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None: #update header table
            headerTable[items[0]][1] = inTree.children[items[0]]

        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])

    if len(items) > 1:#call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:

        if trans in list(retDict.keys()):
            retDict[frozenset(trans)] += 1
        else:
            retDict[frozenset(trans)] = 1

    return retDict

def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)
        
def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)
        if myHead != None:
            #print ('conditional tree for: ',newFreqSet)
            #myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

if __name__ == '__main__':
    tStart = time.time()
    parser = argparse.ArgumentParser(description='dean')
    parser.add_argument('dataset', choices=['kaggle', 'IBM'])
    parser.add_argument("--rules", default=False, action='store_true')
    args = parser.parse_args()

    # hyperparameter
    support_percentage = 0.2

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

    min_sup = len(DB_new) * support_percentage
    DB_new = createInitSet(DB_new)
    myFPtree, myHeaderTab = createTree(DB_new, min_sup)

    freqItems = []
    mineTree(myFPtree, myHeaderTab, min_sup, set([]), freqItems)

    # Helper scripts for getting frequent items and their corresponding supports.
    freqItems_and_sup = {}
    for i in freqItems:
        freqItems_and_sup[str(i)] = 0
        for j in list(DB_new.keys()):
            if i.issubset(j):
                freqItems_and_sup[str(i)] += DB_new.get(j)
    print('FP-Growth\n===')
    for i in freqItems_and_sup.items():
        print("(frequent_item, support)=", i)

    if args.rules:
        freqItems_1 = []
        for i in freqItems:
            freqItems_1.append(i)

        support_only = []
        for i in freqItems_and_sup.values():
            support_only.append(float(i))
        print('\nRules Generation(X -> Y)\n===')
        FP_rules = rules_generation(freqItems_1, support_only)
        print(FP_rules.sort_values(by=['confidence'], ascending=False))

    tEnd = time.time()

    print ("This code cost %f sec." % (tEnd - tStart))
