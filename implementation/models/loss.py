import numpy as np

def entropy_loss(predicted):
  return -1 * np.log(predicted)

def avg_loss(output, expected):
  N = len(output)
  total_loss = 0
  for i in range(N):
    exp_value = expected[i]
    pred_value = output[i][exp_value]
    total_loss += entropy_loss(pred_value)
  return total_loss / N

if __name__ == '__main__':
  output = [[.8, .1, .1], [.8, .1, .1],[.1, .1, .8]]
  expected = [0, 0, 2]
  print avg_loss(output, expected)
