import pickle
import numpy as np
import remeta_v1.remeta as remeta
from remeta_v1.remeta.gendata import simu_data  # noqa
import os
import pathlib
from type2_SDT_MLE import type2_SDT_MLE
from type2roc import type2roc
from scipy.stats import norm
import gzip

# mode = 'default'
# mode = 'type1_only'
# mode = 'type1_complex'
# mode = 'type2_multiplicative_bias'
mode = 'noisy_readout'

def conf(x, bounds):
    confidence = np.full(x.shape, np.nan)
    bounds = np.hstack((bounds, np.inf))
    for i, b in enumerate(bounds[:-1]):
        confidence[(bounds[i] <= x) & (x < bounds[i + 1])] = i + 1
    return confidence
bounds = np.arange(0, 0.81, 0.2)

skip_type2 = False
if mode == 'default':
    nsamples = 2000
    seed = 1
    x_stim_stepsize = 0.25
    params = dict(
        type1_noise=0.5,
        type1_bias=-0.1,
        type2_noise=7,
        type2_criteria=[0.2, 0.2, 0.2, 0.2]
    )
    cfg = remeta.Configuration()
elif mode == 'type1_only':
    nsamples = 2000
    seed = 1
    x_stim_stepsize = 0.25
    params = dict(
        type1_noise=0.7,
        type1_bias=0.2
    )
    cfg = remeta.Configuration()
    skip_type2 = True
elif mode == 'type1_complex':
    nsamples = 2000
    seed = 1
    x_stim_stepsize = 0.02
    params = dict(
        type1_noise=[0.5, 0.7],
        type1_thresh=0.1,
        type1_bias=[0.6, 0.1],
    )
    cfg = remeta.Configuration()
    cfg.enable_type1_param_noise = 2
    cfg.enable_type1_param_thresh = 1
    cfg.enable_type1_param_bias = 2
    skip_type2 = True
elif mode == 'type2_multiplicative_bias':
    nsamples = 2000
    seed = 1
    x_stim_stepsize = 0.25
    params = dict(
        type1_noise=0.6,
        type1_bias=0,
        type2_noise=7,
        type2_evidence_bias_mult=0.8,
    )
    cfg = remeta.Configuration()
    cfg.enable_type2_param_criteria = 0
    cfg.enable_type2_param_evidence_bias_mult = 1
elif mode == 'noisy_readout':
    nsamples = 2000
    seed = 5
    x_stim_stepsize = 0.25
    params = dict(
        type1_noise=0.4,
        type1_bias=0,
        type2_noise=7,
        type2_criteria=[0.3, 0.4, 0.1, 0.1]
    )
    cfg = remeta.Configuration()
    cfg.type2_noise_type = 'noisy_readout'



np.random.seed(seed)
data = simu_data(nsubjects=1, nsamples=nsamples, params=params, cfg=cfg, x_stim_external=None, verbose=True,
                 x_stim_stepsize=x_stim_stepsize, squeeze=True, skip_type2=skip_type2)

stats = dict()
stimulus_ids = (data.x_stim >= 0).astype(int)
correct = (data.d_dec == stimulus_ids).astype(int)
stats['d1'] = norm.ppf(min(1 - 1e-3, max(1e-3, data.d_dec[stimulus_ids == 1].mean()))) - \
              norm.ppf(min(1 - 1e-3, max(1e-3, data.d_dec[stimulus_ids == 0].mean().mean())))
stats['performance'] = np.mean(correct)
stats['choice_bias'] = data.d_dec.mean() - 0.5
if not skip_type2:
    fit = type2_SDT_MLE(stimulus_ids, data.d_dec, conf(data.c_conf, bounds), len(bounds))
    stats['confidence'] = np.mean(data.c_conf)
    stats['auroc2'] = type2roc(correct, data.c_conf)
    stats['mratio'] = fit.M_ratio

path = os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'remeta/demo_data', f'example_data_{mode}.pkl.gz')
save = (data.x_stim, data.d_dec, data.c_conf, params, data.cfg, data.y_decval_mode, stats)
with gzip.open(path, "wb") as f:
    pickle.dump(save, f)
print(f'Saved to {path}')
