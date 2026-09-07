import numpy as np

li = list(range(5))


li2 = [1, True]

li3 = np.array(li)

lis = np.array([[1,2,3], [4,5,6]]) 

from time import time

num_list = [10_000, 1_00_000, 10_00_000, 1_00_00_000]

t_lst = np.array([])
t_arr = np.array([])
for num in num_list:
    lst = list(range(num))
    arr = np.array(lst)
    
    t0 = time()
    lst1 = [i+2 for i in lst]
    t1 = time()
    arr1 = arr+1
    t2 = time()
    t_lst = np.append(t_lst, t1-t0)
    t_arr = np.append(t_arr, t2-t1)