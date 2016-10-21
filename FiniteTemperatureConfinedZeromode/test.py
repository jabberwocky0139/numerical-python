# coding: utf-8

import numpy as np
from scipy.linalg import eig, eig_banded
from pprint import pprint

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10 , 11, 12]])
b = -np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10 , 11, 12]])

print(a)
#print(b)

c = np.array([1, 0, 2])
print(a[c])
