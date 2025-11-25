import numpy as np
from scipy.stats import logistic as logistic_dist
import warnings

try:
    from .configuration import Configuration
    from .dist import get_dist
    from .transform import compute_signal_dependent_type1_noise, type1_evidence_to_confidence, check_criteria_sum
    from .util import _check_param, TAB, type2roc
except ImportError:
    from remeta_v1.remeta.configuration import Configuration
    from remeta_v1.remeta.dist import get_dist
    from remeta_v1.remeta.transform import compute_signal_dependent_type1_noise, type1_evidence_to_confidence, check_criteria_sum
    from remeta_v1.remeta.util import _check_param, TAB, type2roc

class Simulation:
    def __init__(self, nsubjects, nsamples, params, cfg, x_stim, x_stim_category, d_dec, y_decval, y_decval_mode,
                 z1_type1_evidence=None, z1_type1_evidence_mode=None, c_conf=None, c_conf_mode=None,
                 likelihood_dist=None):
        self.nsubjects = nsubjects
        self.nsamples = nsamples
        self.params = params
        self.params_type1 = {k: v for k, v in self.params.items() if k.startswith('type1_')}
        self.params_type2 = {k: v for k, v in self.params.items() if k.startswith('type2_')}
        self.cfg = cfg
        self.x_stim = x_stim
        self.x_stim_category = x_stim_category
        self.d_dec = d_dec
        self.accuracy = x_stim_category == d_dec
        self.y_decval = y_decval
        self.y_decval_mode = y_decval_mode
        self.z1_type1_evidence = z1_type1_evidence
        self.z1_type1_evidence_mode = z1_type1_evidence_mode
        self.c_conf = c_conf
        self.c_conf_mode = c_conf_mode
        self.likelihood_dist = likelihood_dist

    def squeeze(self):
        for var in ('x_stim', 'x_stim_category', 'd_dec', 'accuracy', 'y_decval', 'z1_type1_evidence', 'c_conf'):
            if getattr(self, var) is not None:
                setattr(self, var, getattr(self, var).squeeze())
        return self


def generate_stimuli(nsubjects, nsamples, stepsize=0.02, warn_in_case_of_nondivisible_stepsize=False):
    levels = np.hstack((-np.arange(stepsize, 1.01, stepsize)[::-1], np.arange(stepsize, 1.01, stepsize)))
    if warn_in_case_of_nondivisible_stepsize and ((nsamples % (2/stepsize)) != 0):
        warnings.warn(f'At the chosen stepsize of {stepsize} there are {2/stepsize} stimulus levels,'
                      f'which is not a divisor of the chosen sample size {nsamples}', UserWarning)
    x_stim = np.array([np.random.permutation(np.tile(levels, int(np.ceil(nsamples / len(levels)))))[:nsamples] for _ in range(nsubjects)])
    return x_stim


def simu_type1_responses(x_stim, params, cfg):

    if (cfg.type1_noise_signal_dependency != 'none') or (cfg.enable_type1_param_noise == 2):
        type1_noise = compute_signal_dependent_type1_noise(
            x_stim=x_stim, type1_noise_signal_dependency=cfg.type1_noise_signal_dependency, **params)
    else:
        type1_noise = params['type1_noise']

    type1_param_thresh = _check_param(params['type1_thresh']) if cfg.enable_type1_param_thresh else (0, 0)
    type1_param_bias = _check_param(params['type1_bias']) if cfg.enable_type1_param_bias else (0, 0)

    y_decval_mode = np.full(x_stim.shape, np.nan)
    y_decval_mode[x_stim < 0] = (np.abs(x_stim[x_stim < 0]) > type1_param_thresh[0]) * \
                                   x_stim[x_stim < 0] + type1_param_bias[0]
    y_decval_mode[x_stim >= 0] = (np.abs(x_stim[x_stim >= 0]) > type1_param_thresh[1]) * \
                                    x_stim[x_stim >= 0] + type1_param_bias[1]

    y_decval = y_decval_mode + logistic_dist(scale=type1_noise * np.sqrt(3) / np.pi).rvs(size=x_stim.shape)
    d_dec = (y_decval >= 0).astype(int)

    return y_decval_mode, y_decval, d_dec


def simu_data(nsubjects, nsamples, params, cfg=None, x_stim_external=None, verbose=True, x_stim_stepsize=0.02,
              squeeze=False, force_settings=True, skip_type2=False, warn_in_case_of_nondivisible_stepsize=False, **kwargs):
    params = params.copy()  # this variable can be modifed, thus better to make a copy
    if cfg is None:
        # Set configuration attributes that match keyword arguments
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in Configuration.__dict__}
        cfg = Configuration(force_settings=force_settings, **cfg_kwargs)
        for setting in cfg.__dict__:
            if setting.startswith('enable_'):
                if setting.split('enable_')[1] not in params:
                    setattr(cfg, setting, 0)
    cfg.setup()

    if cfg.type2_noise_dist is None:
        cfg.type2_noise_dist = dict(noisy_report='beta', noisy_readout='gamma')[cfg.type2_noise_type]

    if cfg.type2_noise_dist == 'truncated_norm_transform':
        if cfg.type2_noise_type == 'noisy_report':
            lookup_table = np.load('lookup_truncated_norm_noisy_report.npz')
        elif cfg.type2_noise_type == 'noisy_readout':
            lookup_table = np.load('lookup_truncated_norm_noisy_readout.npz')
    else:
        lookup_table = None

    # Make sure no unwanted parameters have been passed
    for p in ('thresh', 'bias', 'noise_heteroscedastic'):
        if not getattr(cfg, f'enable_type1_param_{p}'):
            params.pop(f'type1_{p}', None)
    for p in ('evidence_bias_mult', 'criteria'):
        if not getattr(cfg, f'enable_type2_param_{p}'):
            params.pop(f'type2_{p}', None)

    if x_stim_external is None:
        x_stim = generate_stimuli(nsubjects, nsamples, stepsize=x_stim_stepsize,
                                  warn_in_case_of_nondivisible_stepsize=warn_in_case_of_nondivisible_stepsize)
    else:
        x_stim = x_stim_external / np.max(np.abs(x_stim_external))
        if x_stim_external.shape != (nsubjects, nsamples):
            x_stim = np.tile(x_stim, (nsubjects, 1))
    x_stim_category = (np.sign(x_stim) > 0).astype(int)
    y_decval_mode, y_decval, d_dec = simu_type1_responses(x_stim, params, cfg)

    if not skip_type2:

        z1_type1_evidence_mode = np.abs(y_decval)

        if cfg.type2_noise_type == 'noisy_readout':
            dist = get_dist(cfg.type2_noise_dist, mode=z1_type1_evidence_mode, scale=params['type2_noise'],
                            log_scale=cfg.type2_noise_logscale,
                            logscale_min=cfg._type2_noise_logscale_min,
                            type2_noise_type=cfg.type2_noise_type, experimental_lookup_table=lookup_table)  # noqa

            z1_type1_evidence = np.maximum(0, dist.rvs((nsubjects, nsamples)))
        else:
            z1_type1_evidence = z1_type1_evidence_mode

        c_conf_mode = type1_evidence_to_confidence(
            z1_type1_evidence=z1_type1_evidence, y_decval=y_decval,
            x_stim=x_stim, type1_noise_signal_dependency=cfg.type1_noise_signal_dependency,
            **params
        )

        if cfg.type2_noise_type == 'noisy_report':
            dist = get_dist(cfg.type2_noise_dist, mode=c_conf_mode, scale=params['type2_noise'],
                            log_scale=cfg.type2_noise_logscale,
                            logscale_min=cfg._type2_noise_logscale_min,
                            type2_noise_type=cfg.type2_noise_type, experimental_lookup_table=lookup_table)
            c_conf = np.maximum(0, np.minimum(1, dist.rvs((nsubjects, nsamples))))
        else:
            c_conf = c_conf_mode

        if cfg.enable_type2_param_criteria or (cfg.type2_fitting_type == 'criteria'):
            if cfg.enable_type2_param_criteria and 'type2_criteria' in params:
                sum_criteria = np.sum(params['type2_criteria'])
                if sum_criteria > 1.001:
                    old_criteria = params['type2_criteria']
                    params['type2_criteria'] = check_criteria_sum(params['type2_criteria'])
                    warnings.warn(
                       '\nThe first entry of the criterion list is a criterion, whereas the subsequent entries encode\n'
                       'the gap to the respective previous criterion. Hence, the sum of all entries in the criterion\n'
                       f'list must be smaller than 1, but sum([{", ".join([f"{c:.3f}" for c in old_criteria])}]) = {sum_criteria:.3f}). '
                       f'Changing criteria to [{", ".join([f"{c:.3f}" for c in params['type2_criteria']])}].', UserWarning)
                first_criterion_and_gaps = params['type2_criteria']
                criteria = [v if i == 0 else np.sum(first_criterion_and_gaps[:i+1]) for i, v in enumerate(first_criterion_and_gaps)]
            else:
                first_criterion_and_gaps = np.ones(cfg.n_discrete_confidence_levels - 1) / cfg.n_discrete_confidence_levels
                criteria = [v if i == 0 else np.sum(first_criterion_and_gaps[:i+1]) for i, v in enumerate(first_criterion_and_gaps)]
                if cfg.enable_type2_param_criteria:
                    warnings.warn(
                        '\nType 2 criteria enabled, but type2_criteria have not been specified. Using default values\n'
                        f'of a Bayesian confidence observer for {cfg.n_discrete_confidence_levels} discrete ratings: [{', '.join([f"{v:.3g}" for v in first_criterion_and_gaps])}].\n'
                        'Note that the first entry of the criterion list is a criterion, whereas the subsequent\n'
                        f'entries encode the gap to the respective previous criterion.\n'
                        f'The final criteria are: [{', '.join([f"{v:.3g}" for v in criteria])}]', UserWarning)

            c_conf = (np.digitize(c_conf, criteria) + 0.5) / cfg.n_discrete_confidence_levels
            c_conf_mode = (np.digitize(c_conf_mode, criteria) + 0.5) / cfg.n_discrete_confidence_levels

    if squeeze:
        x_stim_category = x_stim_category.squeeze()
        x_stim = x_stim.squeeze()
        d_dec = d_dec.squeeze()
        y_decval_mode = y_decval_mode.squeeze()
        y_decval = y_decval.squeeze()
        if not skip_type2:
            z1_type1_evidence_mode = z1_type1_evidence_mode.squeeze()  # noqa
            z1_type1_evidence = z1_type1_evidence.squeeze()  # noqa
            c_conf_mode = c_conf_mode.squeeze()  # noqa
            c_conf = c_conf.squeeze()  # noqa

    simargs = dict(
        nsubjects=nsubjects, nsamples=nsamples, params=params, cfg=cfg,
        x_stim_category=x_stim_category, x_stim=x_stim, d_dec=d_dec,
        y_decval=y_decval, y_decval_mode=y_decval_mode
    )
    if not skip_type2:
        simargs.update(
            z1_type1_evidence=z1_type1_evidence, z1_type1_evidence_mode=z1_type1_evidence_mode,
            c_conf=c_conf, c_conf_mode=c_conf_mode
        )
    simulation = Simulation(**simargs)
    if verbose:
        print('----------------------------------')
        print('Generative model:')
        for p, v in params.items():
            if hasattr(v, '__len__'):
                print(f'{TAB}{p}: [{", ".join([f"{v_:.5g}" for v_ in v])}]')
            else:
                print(f'{TAB}{p}: {v:.5g}')
        if not skip_type2:
            print(f'{TAB}type 2 model: {cfg.type2_noise_type} / {cfg.type2_noise_dist}')
        if 'type2_criteria' in params:
            type2_criteria_absolute = [np.sum(params['type2_criteria'][:i+1]) for i in range(len(params['type2_criteria']))]
            type2_criteria_bias = np.mean(params['type2_criteria'])*(len(params['type2_criteria'])+1)-1
            print(f'{TAB}Type 2 criteria (absolute): [{", ".join([f"{c:.5g}" for c in type2_criteria_absolute])}]')
            print(f'{TAB}Criterion bias: {type2_criteria_bias:.5g}')
        print('----------------------------------')
        print('Basic stats of the simulated data:')
        print(f'{TAB}No. subjects: {nsubjects}')
        print(f'{TAB}No. samples: {nsamples}')
        accuracy = (x_stim_category == d_dec).astype(int)  # noqa
        print(f'{TAB}Performance: {100 * np.mean(accuracy):.1f}% correct')
        choice_bias = 100*d_dec.mean()
        print(f"{TAB}Choice bias: {('-', '+')[int(choice_bias > 50)]}{np.abs(choice_bias - 50):.1f}%")
        if not skip_type2:
            print(f'{TAB}Confidence: {c_conf.mean():.2f}')
            print(f'{TAB}AUROC2: {type2roc(accuracy, c_conf):.2f}')
        print('----------------------------------')

    return simulation


if __name__ == '__main__':
    params_simulation = dict(
        type1_noise=0.2,
        type1_thresh=0.2,
        type1_bias=0.2,
        type2_noise=0.2,
        type2_evidence_bias_mult=1.2
    )
    options = dict(meta_noise_type='noisy_report', enable_type1_param_thresh=1, enable_type1_param_bias=1,
                   enable_type2_param_evidence_bias_mult=1)
    m = simu_data(1, 1000, params_simulation, **options)
