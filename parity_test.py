import numpy as np
from math import ceil

matrix = np.zeros((3,3))
matrix[0,0]=1
matrix[1,1]=1
matrix[2,2]=1
print(matrix[::2,::2])

print(matrix)

list = [ceil(16*(10000/16)**(i/20)) for i in range(21)]

print(list)