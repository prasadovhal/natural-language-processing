import numpy as np

List
Tuple
Set
Dictionary

lst = [1,2,3,4] #list
tpl = (1,2,3,4) # tuple
st = {1,2,3,4} #set


dct = {'a': 1, 'b': 2, 'c': 3, 'd':4}

def add2(lst):
    print(id(lst))
    lst[0] = lst[0]+2
    return lst

def add2():
    print(id(lst))
    lst[0] = lst[0]+2
    return lst

def add2(x,y):
    return x+y

def add2(x=4,y):
    return x+y