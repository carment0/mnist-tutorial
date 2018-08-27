from layers import Softmax
import numpy as np

if __name__ == '__main__':
  layer = Softmax()
  x = np.array([[1.2, 1.1, 0.9, 0.9], [1.05, 0.9, 0.8, 0.9], [0.6, 0.7, 0.5, 0.6]])
  y = np.array([0, 1, 3])
  print 'Brute force way:'
  print layer.backprop(y)
  # print '\nSmart way:'
  # print layer.smart_backprop(x, y)