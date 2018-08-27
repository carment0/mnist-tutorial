import numpy as np

np_arr = np.array([1, 2, 3, 4]).astype('float')

print np_arr / 2

print np.random.rand(4, 4).dot(np.random.rand(4, 4))

# shape is a tuple that gives dimensions of the array.
# If c has n rows and m columns, then c.shape is (n,m)
# >>> c = arange(20).reshape(5,4)
# >>> c
# array([[ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11],
#        [12, 13, 14, 15],
#        [16, 17, 18, 19]])

# >>> c.shape[0]
# >>> 5
# Gives the number of rows

# >>> c.shape[1]
# >>> 4
# Gives number of columns