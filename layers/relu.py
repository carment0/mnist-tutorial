import numpy as np

class ReLU(object):
    def forward(self, x):
        for row in x:
            for idx, val in enumerate(row):
                if val < 0:
                    row[idx] = 0
        return x

if __name__ == '__main__':
    dummy = ReLU()
    x = [[1, -4, 6, -3], [3, -5, -8 , 5]]
    print dummy.forward(x)
