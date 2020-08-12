# HELPER CLASSES & METHODS
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from sklearn.neighbors import KernelDensity
import scipy.stats as stats
from scipy.special import kl_div

import torch.distributions as tdist
import torch.nn.functional as F


def get_y_n(prompt):
    while True:
        try:
            return {"y" or " ": True, "n": False}[input(prompt).lower()]
        except KeyError:
            print("Invalid input---please enter Y or N!")


def ewma(data, alpha, offset=None, dtype=None, order='C', out=None):
    # https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out


def plot_grad_flow(named_parameters):
    # https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


class EarlyStopping(object):
    # from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)


def roll_w_padding(X, X_lengths):
    # input: 2D tensor X, padded w/ zero values to the right
    #        X_lengths: number of nonzero values in each row
    # output: X shifted forwards by 1, maintaining padding
    Xrolled = torch.zeros_like(X)
    for i, x_len in enumerate(X_lengths):
        seq_len = int(X_lengths[i])
        if seq_len:  # should always be nonzero, but still..
            Xrolled[i, :seq_len - 1] = X[i, 1:seq_len]
            if i + 1 < X.shape[0]:  # if not on the last row
                Xrolled[i, seq_len - 1] = X[i + 1, 0]
            else:
                Xrolled[i, seq_len - 1] = X[i, seq_len - 1]
    return Xrolled


def KL_Gauss_np(X):
    # KL divergence of a sequence from the gaussian
    # First the seq is transformed to a probability density function via KDE:
    # https://stackoverflow.com/questions/38711541/how-to-compute-the-probability-of-a-value-given-a-list-of-samples-from-a-distrib
    # Then KL is applied.
    Xpdf = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
    xaxis = np.linspace(min(X)[0], max(X)[0], 1000)[:, np.newaxis]
    kde_vals = np.exp(Xpdf.score_samples(xaxis))
    Xpdf_norm = (kde_vals - kde_vals.min()) / (kde_vals - kde_vals.min()).sum()

    unit_gauss = stats.norm.pdf(xaxis, np.mean(X))[:, 0]

    return kl_div(Xpdf_norm, unit_gauss).sum()


"""
# TEST
print(KL_unitGauss(torch.DoubleTensor([[-1], [-1], [1], [1], [2], [2]])))

#print(KL_unitGauss([[-1], [0], [1], [1], [1], [1],  [2], [3]]))
"""
