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
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        learning training data sets

        Arguments
        --------------
        X: { array-like }, shape = [n_samples, n_features]

        y: { array-like }, shape = [n_samples] => target value

        returning value
        -----------------
        self: object
        """
        rgen = np.random.RandomState(self.random_state)
        """
        평균이 0 이고, 표준편차가 0.01 인 정규분포를 따르는 난수 생성
        """
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1]) # initialize weight
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi)) # update weight
                self.w_[1:] += update * xi # 새로운 연결강도 = 현재 연결강도 + 학습률 * (실제값(ground truth) - 예측값)
                self.w_[0] += update # 결정 경계를 보정하기 위함
                print("test", self.w_[0])
                errors += int(update != 0.0) # 오차가 있을경우 (즉, 0 보다 클 경우)
            self.errors_.append(errors)
        return self


    def net_input(self, X):
        """ calculate input value """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        
        