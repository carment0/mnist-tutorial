from time import sleep
import math

data = [(0, 100), (1, 150), (2, 200), (3, 250)]
params = (50, 100) #answer
housing_data = [
(2.275, 3.891),
(3.472, 4.5573),
(3.531, 4.9176),
(4, 5.0208),
(4.05, 5.05),
(4.455, 5.0597),
(4.455, 5.3003),
(4.9883, 5.6039),
(5, 5.6039)
]

test_data = [
(5.0, 5.8282),
(5.15, 5.898),
(5.52, 5.9592),
(5.52, 5.9894),
(5.85, 6.0931),
(6.435, 6.2712),
(6.666, 6.6969),
(6.7265, 7.5422),
(6.902, 7.7841),
(7.102, 8.14),
(7.3262, 8.2464),
(7.8, 8.3607),
(8, 8.7951),
(9.15, 9.0384),
(9.52, 9.1415),
(9.8, 9.8598),
(9.89, 9.998),
(12.8, 10.4202)
]

# slope formula use bc dataset is linear. more bedroom, more money
def pred(x, params):
    p0, p1 = params
    return p0 * x + p1

def loss(data, pred, params):
    loss = 0
    for datum in data:
        x, y = datum
        loss += (y - pred(x, params))**2

    return loss

# loss(data, pred, params)

# Slope is rise over run, and derivative is slope
# slope is a approx, while derivative is real value
# if you have a curved line you can find the "actual slope" by two ways
# the rise over run way (when you zoom into a curve line, eventually the curve
# will be a line) or taking the derivative

def derivative_respect_to_p0(data, p0, p1):
    slope = 0
    for datum in data:
        x, y = datum
        slope += 2 * (p0 * x + p1 - y) * x

    return slope

def derivative_respect_to_p1(data, p0, p1):
    slope = 0
    for datum in data:
        x, y = datum
        slope += 2 * (p0 * x + p1 - y)

    return slope

# rise = loss(data, pred, params=(2.00001, 4)) - loss(data, pred, params=(2, 4))
# run = 2.00001 - 2
# print "Slope is %f" % (rise/run)
# print "Derivative is %f" %derivative_respect_to_p0(data, 2, 4)

# rise = loss(data, pred, params=(2, 4.00001)) - loss(data, pred, params=(2, 4))
# run = 4.00001 - 4
# print "Slope is %f" % (rise/run)
# print "Derivative is %f" %derivative_respect_to_p1(data, 2, 4)


def test_model(test_data, params):
  """Root Mean Squared Error
  https://en.wikipedia.org/wiki/Root-mean-square_deviation
  """
  err = 0
  for datum in test_data:
    x_test, y_test = datum
    my_model_prediction = pred(x_test, params)
    err += (y_test - my_model_prediction)**2

  return math.sqrt(err) / len(test_data)

# Each P is one dimensions, gradient descent (rolling) is a finding the lowest point in multiple dimensions slope
# Perform gradient descent
params = (0,0) #random initial values
step_size = 0.0001 #how much you want to "roll"
count = 1
while count <= 10000:
    p0, p1 = params
    print 'Current loss %f with p0 = %f and p1 = %f' % (loss(housing_data, pred, params), p0, p1)
    d_p1 = derivative_respect_to_p1(housing_data, p0, p1)
    d_p0 = derivative_respect_to_p0(housing_data, p0, p1)

    p0 = p0 - step_size * d_p0
    p1 = p1 - step_size * d_p1

    params = (p0, p1)
    count += 1
RMSD = test_model(test_data, params)
print 'The RMSE is %f' % (RMSD)
# when you run this, you will see the loss will decrease to 0, p0 going to 50 and p1 going to 100


# https://en.wikipedia.org/wiki/Linear_regression