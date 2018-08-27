import numpy as np

x = np.array([[1,2,3,4]])
y = np.array([[1,1,1], [1,1,1], [1,1,1], [1,1,1]])
b = np.array([[5, 6, 7]])
print np.dot(x, y) + b