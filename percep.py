import numpy as np

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

x = np.array([0, 1]) # input
w = np.array([0.5, 0.5]) # weight
b = -0.7 # bias

print(np.sum(w * x) + b)