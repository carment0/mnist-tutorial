# MNIST Tutorial
## Goals
1. Use `tensorflow` and `keras` to build a simple fully connected neural network to perform hand written digit classification on MNIST dataset.
2. Reimplement the functionality of `keras` (neural networks) from stratch

## Progress
1. Built a simple sequential model using `keras` [Keras MNIST Example.ipynb]
2. Implement from scratch (forward propagation)
  - Dense layer
  - ReLU Activation
  - Softmax Activation
  - Cross Entropy Loss

## Python Environment
* `pip`
* `virtualenv`
- `virtualenv environment`
- `source environment/bin/activate`

## Jupyter Notebook
* `jupyter`
- run after you activate your env `jupyter notebook`

## Testing
* `pip install nose`
- run test `nosetests`
- run test with logs `nosetests --nocapture`

## `__init__.py`
This is the file where you export your function and classes
It is equivalent to a constructor for a package.


_________________________________________________
Chain rule
f(x) = sin(cos(3x))
  h(x) = 3x
  g(h) = cos(h)
  f(g) = sin(g)
    1. grad_f w.r.t g *TIMES*
    2. grad_g w.r.t h *TIMES*
    3. grad_h w.r.t x

this is equal to:
softmax(dense(x)) => answer or desired output of your neural network
  dense(x) => s
  softmax(s) => probability
    1. grad_softmax w.r.t score *times*
    2. grad_dense w.r.t x

e.g. for 6 layers
dense_1(x)
relu_1(dense_1)
dense_2(relu_1)
relu_2(dense_2)
dense_3(relu_2)
softmax(dense_3)
