import numpy as np

matrix = np.zeros((3,3))
matrix[0,0]=1
matrix[1,1]=1
matrix[2,2]=1
print(matrix[::2,::2])

matrix[::2,::2] = np.array([2,2,2,2,2])

print(matrix)