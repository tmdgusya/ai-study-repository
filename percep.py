import numpy as np
import matplotlib.pylab as plt

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

def AND_WITH_BIAS(x1, x2):
    w1, w2, bias = 0.5, 0.5, -0.7
    tmp = x1 * w1 + x2 * w2 + bias
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    result = AND(s1, s2)
    return result

def step_function(x):
    y = x > 0
    return y.astype(np.int_)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

print(AND_WITH_BIAS(0, 0))
print(AND_WITH_BIAS(0, 1))
print(AND_WITH_BIAS(1, 0))
print(AND_WITH_BIAS(1, 1))

print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))

print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))

print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)


x = np.array([0, 1]) # input
w = np.array([0.5, 0.5]) # weight
b = -0.7 # bias

print(np.sum(w * x) + b)

x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y 축 범위 지정
plt.show()