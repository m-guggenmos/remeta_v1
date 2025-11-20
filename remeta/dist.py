import warnings

import numpy as np
from scipy.stats import norm, lognorm, beta, betaprime, gamma, invgamma, uniform, gumbel_r

try:
    from .util import maxfloat
    from .fast_truncnorm import truncnorm
except:
    from remeta_v1.remeta.util import maxfloat
    from remeta_v1.remeta.fast_truncnorm import truncnorm

TYPE2_NOISE_DISTS = (
    'beta', 'beta_std',
    'norm', 'gumbel', 'lognorm', 'lognorm_varstd', 'betaprime', 'gamma', 'invgamma',
    'truncated_norm', 'truncated_norm_lookup', 'truncated_norm_fit',
    'truncated_gumbel', 'truncated_gumbel_lookup',
    'truncated_lognorm', 'truncated_lognorm_varstd'
)
TYPE2_NOISE_DISTS_REPORT_ONLY = ('beta', 'beta_std')
TYPE2_NOISE_DISTS_READOUT_ONLY = ('lognorm', 'lognorm_varstd', 'betaprime', 'gamma')


def _lognorm_params(mode, stddev):
    """
    Compute scipy lognorm's shape and scale for a given mode and SD
    The analytical formula is exact and was computed with WolframAlpha.

    Parameters
    ----------
    mode : float or array-like
        Mode of the distribution.
    stddev : float or array-like
        Standard deviation of the distribution.

    Returns
    ----------
    shape : float
        Scipy lognorm shape parameter.
    scale : float
        Scipy lognorm scale parameter.
    """
    mode = np.maximum(1e-5, mode)
    a = stddev ** 2 / mode ** 2
    x = 1 / 4 * np.sqrt(np.maximum(1e-300, -(16 * (2 / 3) ** (1 / 3) * a) / (
                np.sqrt(3) * np.sqrt(256 * a ** 3 + 27 * a ** 2) - 9 * a) ** (1 / 3) +
                                   2 * (2 / 3) ** (2 / 3) * (
                                               np.sqrt(3) * np.sqrt(256 * a ** 3 + 27 * a ** 2) - 9 * a) ** (
                                               1 / 3) + 1)) + \
        1 / 2 * np.sqrt(
        (4 * (2 / 3) ** (1 / 3) * a) / (np.sqrt(3) * np.sqrt(256 * a ** 3 + 27 * a ** 2) - 9 * a) ** (1 / 3) -
        (np.sqrt(3) * np.sqrt(256 * a ** 3 + 27 * a ** 2) - 9 * a) ** (1 / 3) / (2 ** (1 / 3) * 3 ** (2 / 3)) +
        1 / (2 * np.sqrt(np.maximum(1e-300, -(16 * (2 / 3) ** (1 / 3) * a) / (
                    np.sqrt(3) * np.sqrt(256 * a ** 3 + 27 * a ** 2) - 9 * a) ** (1 / 3) +
                                    2 * (2 / 3) ** (2 / 3) * (
                                                np.sqrt(3) * np.sqrt(256 * a ** 3 + 27 * a ** 2) - 9 * a) ** (
                                                1 / 3) + 1))) + 1 / 2) + \
        1 / 4
    shape = np.sqrt(np.log(x))
    scale = mode * x
    return shape, scale


class truncated_lognorm:  # noqa
    """
    Implementation of the truncated lognormal distribution.
    Only the upper truncation bound is supported as the lognormal distribution is naturally lower-bounded at zero.

    Parameters
    ----------
    loc : float or array-like
        Scipy lognorm's loc parameter.
    scale : float or array-like
        Scipy lognorm's scale parameter.
    s : float or array-like
        Scipy lognorm's s parameter.
    b : float or array-like
        Upper truncation bound.
    """
    def __init__(self, loc, scale, s, b):
        self.loc = loc
        self.scale = scale
        self.s = s
        self.b = b
        self.dist = lognorm(loc=loc, scale=scale, s=s)
        self.lncdf_b = self.dist.cdf(self.b)

    def pdf(self, x):
        pdens = (x <= self.b) * self.dist.pdf(x) / self.lncdf_b
        return pdens

    def cdf(self, x):
        cdens = (x > self.b) + (x <= self.b) * self.dist.cdf(x) / self.lncdf_b
        return cdens

    def rvs(self, size=None):
        if size is None:
            if hasattr(self.scale, '__len__'):
                size = self.scale.shape
            else:
                size = 1
        cdens = uniform(loc=0, scale=self.b).rvs(size)
        x = self.cdf_inv(cdens)
        return x

    def cdf_inv(self, cdens):
        x = (cdens >= 1) * self.b + (cdens < 1) * self.dist.ppf(cdens * self.lncdf_b)
        return x


class truncated_gumbel:  # noqa
    """
    Implementation of the truncated Gumbel distribution.

    Parameters
    ----------
    loc : float or array-like
        Scipy gumbel_r's loc parameter.
    scale : float or array-like
        Scipy gumbel_r's scale parameter.
    a : float or array-like
        Lower truncation bound.
    b : float or array-like
        Upper truncation bound.
    """
    def __init__(self, loc, scale, a=0, b=np.inf):
        self.loc = loc
        self.scale = scale
        self.a = a
        self.b = b
        self.dist = gumbel_r(loc=loc, scale=scale)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            self.cdf_to_a = self.dist.cdf(self.a)
        self.cdf_a_to_b = self.dist.cdf(self.b) - self.cdf_to_a

    def pdf(self, x):
        pdens = ((x > self.a) & (x < self.b)) * self.dist.pdf(maxfloat(x)).astype(np.float64) / self.cdf_a_to_b
        return pdens

    def cdf(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cdf_to_x = self.dist.cdf(x)
        cdens = (x >= self.b) + ((x > self.a) & (x < self.b)) * (cdf_to_x - self.cdf_to_a) / self.cdf_a_to_b
        return cdens

    def rvs(self, size=None):
        if size is None:
            if hasattr(self.scale, '__len__'):
                size = self.loc.shape
            else:
                size = 1
        cdens = uniform(loc=self.cdf_to_a, scale=self.cdf_a_to_b).rvs(size)
        x = self.cdf_inv(cdens)
        return x

    def cdf_inv(self, cdens):
        x = (cdens > self.cdf_to_a) * self.dist.ppf(cdens)
        return x


def get_dist(type2_dist, mode, scale, type2_noise_type='noisy_report', lookup_table=None):
    """
    Helper function to select appropriately parameterized metacognitive noise distributions.
    """

    if type2_dist not in TYPE2_NOISE_DISTS:
        raise ValueError(f"Unkonwn distribution '{type2_dist}'.")
    elif (type2_noise_type == 'noisy_report') and type2_dist in TYPE2_NOISE_DISTS_READOUT_ONLY:
        raise ValueError(f"Distribution '{type2_dist}' is only valid for noisy-readout models.")
    elif (type2_noise_type == 'noisy_readout') and type2_dist in TYPE2_NOISE_DISTS_REPORT_ONLY:
        raise ValueError(f"Distribution '{type2_dist}' is only valid for noisy-report models.")

    if type2_dist == 'norm':
        dist = norm(loc=mode, scale=scale)
    elif type2_dist == 'gumbel':
        dist = gumbel_r(loc=mode, scale=scale * np.sqrt(6) / np.pi)
    elif type2_dist == 'lognorm_varstd':
        dist = lognorm(loc=0, scale=np.maximum(1e-5, mode) * np.exp(scale ** 2), s=scale)
    elif type2_dist == 'lognorm':
        shape, scale = _lognorm_params(np.maximum(1e-5, mode), scale)
        dist = lognorm(loc=0, scale=scale, s=shape)
    elif type2_dist == 'lognorm_varstd':
        dist = lognorm(loc=0, scale=np.maximum(1e-5, mode) * np.exp(scale ** 2), s=scale)
    elif type2_dist == 'beta':
        a = mode * (1 / scale - 2) + 1
        b = (1 - mode) * (1 / scale - 2) + 1
        dist = beta(a, b)
    elif type2_dist == 'beta_std':
        mode = np.maximum(1e-5, np.minimum(1-1e-5, mode))
        a = 1 + (1 - mode) * mode**2 / scale**2
        b = (1/mode - 1) * a - 1/mode + 2
        dist = beta(a, b)
    elif type2_dist == 'betaprime':
        mode = np.minimum(1 / scale - 1 - 1e-3, mode)
        a = (mode * (1 / scale + 1) + 1) / (mode + 1)
        b = (1 / scale - mode - 1) / (mode + 1)
        dist = betaprime(a, b)
    elif type2_dist == 'gamma':
        a = (mode + np.sqrt(mode**2 + 4*scale**2))**2 / (4*scale**2)
        b = (mode + np.sqrt(mode**2 + 4*scale**2)) / (2*scale**2)
        dist = gamma(a=a, loc=0, scale=1/b)
    elif type2_dist == 'invgamma':
        r = scale**2 / mode**2
        a = (2*r+1 + np.sqrt((2*r+1)**2+8*r)) / (2*r)
        b = mode*(a+1)
        dist = invgamma(a=a, loc=0, scale=b)
    elif type2_dist.startswith('truncated_'):
        if type2_noise_type == 'noisy_report':
            if type2_dist.endswith('_lookup') and (type2_dist.startswith('truncated_norm_') or
                                                 type2_dist.startswith('truncated_gumbel_')):
                m_ind = np.searchsorted(lookup_table['mode'], mode)
                scale = lookup_table['scale'][np.abs(lookup_table['truncscale'][m_ind] - scale).argmin(axis=-1)]
            elif type2_dist == 'truncated_norm_fit':
                mode_ = mode.copy()
                mode_[mode_ < 0.5] = 1 - mode_[mode_ < 0.5]
                scale = np.minimum(scale, 1/np.sqrt(12) - 1e-3)
                alpha1, beta1, alpha2, beta2, theta = -0.1512684, 4.15388204, -1.01723445, 2.47820677, 0.73799941
                scale = scale / (beta1*mode_**alpha1*np.sqrt(1/12-scale**2)) * (mode_ < theta) + \
                    (np.arctanh(2*np.sqrt(3)*scale) / (beta2*mode_**alpha2)) * (mode_ >= theta)
            if type2_dist.startswith('truncated_norm'):
                dist = truncnorm(-mode / scale, (1 - mode) / scale, loc=mode, scale=scale)
            elif type2_dist.startswith('truncated_gumbel'):
                dist = truncated_gumbel(loc=mode, scale=scale * np.sqrt(6) / np.pi, b=1)
            elif type2_dist == 'truncated_lognorm':
                shape, scale = _lognorm_params(np.maximum(1e-5, mode), scale)
                dist = truncated_lognorm(loc=0, scale=scale, s=shape, b=1)
            elif type2_dist == 'truncated_lognorm_varstd':
                dist = truncated_lognorm(loc=0, scale=np.maximum(1e-5, mode) * np.exp(scale ** 2), s=scale, b=1)
        elif type2_noise_type == 'noisy_readout':
            if type2_dist.endswith('_lookup') and (type2_dist.startswith('truncated_norm_') or
                                                 type2_dist.startswith('truncated_gumbel_')):
                m_ind = np.searchsorted(lookup_table['mode'], np.minimum(10, mode))  # atm 10 is the maximum mode
                scale = lookup_table['scale'][np.abs(lookup_table['truncscale'][m_ind] - scale).argmin(axis=-1)]
            elif type2_dist == 'truncated_norm_fit':
                a, b, c, d, e, f = 0.88632051, -1.45129289, 0.25329918, 2.09066054, -1.2262868, 1.47179606
                scale = a*scale*(mode+1)**b + c*(mode+1)**f*(np.exp(np.minimum(100, d*scale*(mode+1)**e))-1)
            if type2_dist.startswith('truncated_norm'):
                dist = truncnorm(-mode / scale, np.inf, loc=mode, scale=scale)
            elif type2_dist.startswith('truncated_gumbel'):
                dist = truncated_gumbel(loc=mode, scale=scale * np.sqrt(6) / np.pi)
        else:
            raise ValueError(f"'{type2_noise_type}' is an unknown metacognitive type")

    return dist  # noqa


def get_dist_mean(distname, mode, scale):
    """
    Helper function to get the distribution mean of certain distributions.
    """
    if distname == 'gumbel':
        mean = mode + np.euler_gamma * scale * np.sqrt(6) / np.pi
    elif distname == 'norm':
        mean = mode
    else:
        raise ValueError(f"Distribution {distname} not supported.")
    return mean


def get_pdf(x, distname, mode, scale, lb=0, ub=1, meta_noise_type='noisy_report'):
    """
    Helper function to get the probability density of a distribution.
    """
    return get_dist(distname, mode, scale, meta_noise_type).pdf(x)


def get_likelihood(x, distname, mode, scale, binsize_meta=1e-3, logarithm=False):
    """
    Helper function to get the likelihood of a distribution.
    """
    dist = get_dist(distname, mode, scale)
    likelihood = dist.cdf(x + binsize_meta) - dist.cdf(x - binsize_meta)
    return np.log(np.maximum(1e-4, likelihood)) if logarithm else likelihood
