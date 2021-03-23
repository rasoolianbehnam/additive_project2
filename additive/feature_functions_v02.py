from scipy.stats import mode
import numpy as np


def ra_1d_(x):
    return np.mean(np.abs(x - np.mean(x, axis=1, keepdims=True)), axis=1)


def rq_1d_(x):
    return np.sqrt(np.mean((x - np.mean(x, axis=1, keepdims=True)) ** 2, axis=1))


class Features:
    def __init__(self, aggregator_id=np.mean):
        self.aggregator_1d = aggregator_id

    @property
    def functions(self):
        out = {k.replace('feature_', ''): getattr(self, k) for k in dir(self) if k.startswith('feature')}
        out['median'] = np.median
        out['mean'] = np.mean
        return out

    def feature_ra_1d(self, x):
        return self.aggregator_1d(ra_1d_(x))

    def feature_rq_1d(self, x):
        return self.aggregator_1d(rq_1d_(x))

    def feature_rsk_1d(self, x):
        rq = rq_1d_(x)
        return self.aggregator_1d(np.mean((x - np.mean(x, axis=1, keepdims=True)) ** 3, axis=1) / rq ** 3)

    def feature_rku_1d(self, x):
        rq = rq_1d_(x)
        return self.aggregator_1d(np.mean((x - np.mean(x, axis=1, keepdims=True)) ** 4, axis=1) / rq ** 4)

    def feature_rp_1d(self, x):
        return self.aggregator_1d(np.max(x - np.mean(x, axis=1, keepdims=True), axis=1))

    def feature_rv_1d(self, x):
        return -self.aggregator_1d(np.min(x - np.mean(x, axis=1, keepdims=True), axis=1))

    def feature_ra_2d(self, x):
        return np.mean(np.abs(x - np.mean(x)))

    def feature_rq_2d(self, x):
        return np.sqrt(np.mean((x - np.mean(x)) ** 2))

    def feature_rp_2d(self, x):
        return np.max(x - np.mean(x))

    def feature_rv_2d(self, x):
        return -np.min(x - np.mean(x))

    def feature_rsk_2d(self, x):
        rq = self.feature_rq_2d(x)
        return np.mean((x - np.mean(x)) ** 3) / rq ** 3

    def feature_rku_2d(self, x):
        rq = self.feature_rq_2d(x)
        return np.mean((x - np.mean(x)) ** 4) / rq ** 4

    def feature_mode_1d(self, x):
        modes = mode((x - x.mean()).round(decimals=0), axis=1)
        return modes.mode.mean()

    def feature_mode_2d(self, x, bins=100):
        freq, val = np.histogram(x - x.mean(), bins=bins, density=True, )
        return val[np.argmax(freq)]

