'''github: https://github.com/takoika/PrioritizedExperienceReplay/blob/master/sum_tree.py'''

import sys
import os
import math


class SumTree(object):

    '''
    max_size: max number of elements in tree
    tree_level: how many levels the tree has
    tree_size: how many nodes in the tree
    tree: list representation of tree
    data: data content stored in sliding window
    size: current number of values in tree
    cursor: index of next empty value in list, loops back when list is full

    '''
    def __init__(self, max_size):
        self.max_size = max_size
        self.tree_level = math.ceil(math.log(max_size+1, 2))+1
        self.tree_size = 2**self.tree_level-1
        self.tree = [0 for i in range(self.tree_size)]
        self.data = [None for i in range(self.max_size)]
        self.size = 0
        self.cursor = 0

    '''
    contents: content to be stored in data
    value: value assosciated with contents
    '''
    def add(self, contents, value):
        index = self.cursor
        # cursor loops back when list is full
        self.cursor = (self.cursor+1)%self.max_size
        self.size = min(self.size+1, self.max_size)

        self.data[index] = contents
        self.val_update(index, value)

    def get_val(self, index):
        tree_index = 2**(self.tree_level-1)-1+index
        return self.tree[tree_index]

    def val_update(self, index, value):
        tree_index = 2**(self.tree_level-1)-1+index
        diff = value-self.tree[tree_index]
        self.reconstruct(tree_index, diff)

    def reconstruct(self, tindex, diff):
        self.tree[tindex] += diff
        if not tindex == 0:
            tindex = int((tindex-1)/2)
            self.reconstruct(tindex, diff)

    '''
    if norm=True, value should be [0,1]
    '''
    def find(self, value, norm=True):
        if norm:
            value *= self.tree[0]
        return self._find(value, 0)

    def _find(self, value, index):
        if 2**(self.tree_level-1)-1 <= index:
            # returns [ content, value, index ]
            return self.data[index-(2**(self.tree_level-1)-1)], self.tree[index], index-(2**(self.tree_level-1)-1)

        left = self.tree[2*index+1]

        if value <= left:
            return self._find(value,2*index+1)
        else:
            return self._find(value-left,2*(index+1))
        
    def print_tree(self):
        for k in range(1, self.tree_level+1):
            for j in range(2**(k-1)-1, 2**k-1):
                print(self.tree[j], end=' ')
            print()

    def filled_size(self):
        return self.size

if __name__ == '__main__':
    s = SumTree(10)
    for i in range(20):
        s.add((2**i,i**2), i)
    s.print_tree()
    print(s.find(0.5))

'''
145 
108 37 
46 62 37 0 
21 25 29 33 37 0 0 0 
10 11 12 13 14 15 16 17 18 19 0 0 0 0 0 0   # last row has 10 values + 16-10=6 zeros, sliding window so only last 10 are stored
(32768, 15, 5)
'''



