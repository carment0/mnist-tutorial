from time import sleep

data = [(0, 100), (1, 150), (2, 200), (3, 250)]
params = (50, 100) #answer

# slope formula use bc dataset. more bedroom, more money
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

rise = loss(data, pred, params=(2.00001, 4)) - loss(data, pred, params=(2, 4))
run = 2.00001 - 2
print "Slope is %f" % (rise/run)
print "Derivative is %f" %derivative_respect_to_p0(data, 2, 4)

rise = loss(data, pred, params=(2, 4.00001)) - loss(data, pred, params=(2, 4))
run = 4.00001 - 4
print "Slope is %f" % (rise/run)
print "Derivative is %f" %derivative_respect_to_p1(data, 2, 4)

# Each P is one dimensions, gradient descent (rolling) is a finding the lowest point in multiple dimensions slope
# Perform gradient descent
params = (0,0) #random initial values
step_size = 0.0001 #how much you want to "roll"
while True:
    p0, p1 = params
    print 'Current loss %f with p0 = %f and p1 = %f' % (loss(data, pred, params), p0, p1)
    d_p1 = derivative_respect_to_p1(data, p0, p1)
    d_p0 = derivative_respect_to_p0(data, p0, p1)

    p0 = p0 - step_size * d_p0
    p1 = p1 - step_size * d_p1

    params = (p0, p1)
    sleep(0.2)
# when you run this, you will see the loss will decrease to 0, p0 going to 50 and p1 going to 100
