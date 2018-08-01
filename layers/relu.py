import numpy as np

class ReLU(object):
    def forward(self, x):
        x[x < 0] = 0
        return x

if __name__ == '__main__':
    dummy = ReLU()
    x = [[1, -4, 6, -3], [3, -5, -8 , 5]]
    arr = np.array(x)
    print dummy.forward(arr)
