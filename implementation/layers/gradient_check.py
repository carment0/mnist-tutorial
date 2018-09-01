import numpy as np

# Let's create a numerical gradient!
# Lambda in Python is equivalent to arrow function in Javascipt
def eval_numerical_gradient(f, x, delta=1e-5):
  """Evaluate gradient of a function numerically

  Args:
    f (funcation): This is a callback that you invoke with x
    x (numpy.ndarray): A matrix that serves as an input to f
    delta (float): A small float number for calculating slope i.e. gradient

  Returns:
    grad (numpy.ndarray): Gradient of f with respect to x
  """
  grad = np.zeros_like(x) # Set up a gradient matrix, of the same shape as input

  itr = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not itr.finished:
    idx = itr.multi_index
    # print x[idx]

    init_val = x[idx]
    x[idx] = init_val + delta
    f_plus = f(x)

    x[idx] = init_val - delta
    f_minus = f(x)

    x[idx] = init_val

    # compute slope
    grad[idx] = (f_plus - f_minus) / (2 * delta)

    itr.iternext()

  return grad

# def foo(x):
#   return x + 1

# f = lambda x: foo(x)

# rand_input = np.random.rand(10, 10, 10)
# eval_numerical_gradient(f, rand_input)

# # def bar(x):
# #   return x * 2

# # # foo() will be called first than bar()
# # f = lambda x: bar(foo(x))
# # # Equivalent to const f = (x) => foo(x)
# # # Equivalent to f = Proc.new{|x| foo(x)}
# # # f = lambda x: avg_loss(self.forward_prop(x))

# # random_input = np.array([[1,1,1], [1,1,1], [1,1,1]])
# # print f(random_input)

def eval_numerical_gradient_for_matrix(f, x, grad_f, delta=1e-5):
  """Evaluate gradient numerically for a funcation that returns a matrix as output, e.g. Dense
  """
  grad = np.zeros_like(x)
  itr = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not itr.finished:
    idx = itr.multi_index

    init_val = x[idx]

    x[idx] = init_val + delta
    f_plus = f(x)

    x[idx] = init_val - delta
    f_minus = f(x)

    x[idx] = init_val

    # compute slope
    grad[idx] = np.sum((f_plus - f_minus) * grad_f) / (2 * delta)

    itr.iternext()

  return grad
