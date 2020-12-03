import numpy as np

x = np.array([1,2,3])
y = np.array([4,5,6])

for i,j in zip(x, y):
	print(i, j)