import numpy as np

def log_gaussian(X, mean, var):
    # log of the Gaussian density, per value
    return -0.5 * np.log(2 * np.pi * var) - (X - mean) ** 2 / (2 * var)

class MyGaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)                       # sorted class names
        n_classes = len(self.classes_)
        self.epsilon_ = self.var_smoothing * X.var(axis=0).max()  # tiny safety number
        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]                                # rows of this class only
            self.priors_[i] = X_c.shape[0] / n_samples
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + self.epsilon_
        return self

    def log_scores(self, X):
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i in range(len(self.classes_)):
            log_prior = np.log(self.priors_[i])
            log_like = log_gaussian(X, self.means_[i], self.vars_[i]).sum(axis=1)
            scores[:, i] = log_prior + log_like
        return scores

    def predict(self, X):
        best = np.argmax(self.log_scores(X), axis=1)       # winning column per row
        return self.classes_[best]
