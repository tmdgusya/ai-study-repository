import numpy as np

class Perceptron(object):
    """

    Arguments
    ---------------
    eta: float
        learning late (between 0.0 and 1.0)
    n_iter: int
        iteration count for learning
    random_state: int
        for generating random seed to use it as weight


    Properties
    -----------------
    w_ : 1d-array
        weight trained
    errors_ : list
        missed value for each epoch

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self_n_iter = n_iter