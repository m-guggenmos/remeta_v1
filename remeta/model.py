import os
import pathlib
import pickle
import timeit
import warnings
from dataclasses import make_dataclass

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import logistic as logistic_dist, binom

try:
    from .configuration import Configuration
    from .dist import get_dist
    from .fit import fmincon
    from .gendata import simu_data
    from .modelspec import Model, Data
    from .plot import plot_evidence_versus_confidence, plot_confidence_dist
    from .transform import compute_signal_dependent_type1_noise, logistic, type1_evidence_to_confidence, confidence_to_type1_evidence, check_criteria_sum
    from .util import _check_param, TAB
    from .util import maxfloat
except ImportError:
    from remeta_v1.remeta.configuration import Configuration
    from remeta_v1.remeta.dist import get_dist
    from remeta_v1.remeta.fit import fmincon
    from remeta_v1.remeta.gendata import simu_data
    from remeta_v1.remeta.modelspec import Model, Data
    from remeta_v1.remeta.plot import plot_evidence_versus_confidence, plot_confidence_dist
    from remeta_v1.remeta.transform import compute_signal_dependent_type1_noise, logistic, type1_evidence_to_confidence, confidence_to_type1_evidence, check_criteria_sum
    from remeta_v1.remeta.util import _check_param, TAB
    from remeta_v1.remeta.util import maxfloat

np.set_printoptions(suppress=True)


class ReMeta:

    def __init__(self, cfg=None, force_settings=False, **kwargs):
        """
        Main class of the ReMeta toolbox

        Parameters
        ----------
        cfg : util.Configuration
            Configuration object. If None is passed, the default configuration is used (but see kwargs).
        force_settings : bool (default: False)
            Force settings as specified in the configuration instance
        kwargs : dict
            The kwargs dictionary is parsed for keywords that match keywords of util.Configuration; in case of a match,
            the configuration is set.
        """

        if cfg is None:
            # Set configuration attributes that match keyword arguments
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in Configuration.__dict__}
            self.cfg = Configuration(**cfg_kwargs)
        else:
            self.cfg = cfg
        self.cfg.setup(force_settings=force_settings)

        if self.cfg.type2_noise_dist.startswith('truncated_') and self.cfg.type2_noise_dist.endswith('_lookup'):
            try:
                self.lookup_table = np.load(f"lookup_{self.cfg.type2_noise_dist}_{self.cfg.type2_noise_type}.npz")
            except FileNotFoundError:
                raise FileNotFoundError('Lookup table not found. Lookup tables are not deployed via pip. You can '
                                        'download them from Github and put in a directory named "lookup"')
        else:
            self.lookup_table = None

        self.model = Model(cfg=self.cfg)
        self.data = None

        # self._punish_message = False

        self.type1_is_fitted = False
        self.type2_is_fitted = False

        # Negative log likleihood function that is minimized for type 2 parameter fitting
        self.fun_negll_type2 = dict(noisy_report=self._negll_type2_noisyreport,
                                    noisy_readout=self._negll_type2_noisyreadout)[self.cfg.type2_noise_type]
        self.fun_negll_type2_helper = dict(noisy_report=self._helper_negll_type2_noisyreport,
                                           noisy_readout=self._helper_negll_type2_noisyreadout)[self.cfg.type2_noise_type]

    def fit(self, x_stim, d_dec, c_conf, precomputed_parameters=None, guess_type2=None, verbose=True,
            ignore_warnings=False, skip_type2=False):
        """
        Fit type 1 and type 2 parameters

        Parameters
        ----------
        x_stim : array-like of shape (n_samples)
            Array of signed stimulus intensity values, where the sign codes the stimulus category (cat 1: -, cat2: +)
            and the absolut value codes the intensity. Must be normalized to [-1; 1], or set
            `normalize_stimuli_by_max=True`.
        d_dec : array-like of shape (n_samples)
            Array of choices coded as 0 (or alternatively -1) for the negative stimuli category and 1 for the positive
            stimulus category.
        c_conf : array-like of shape (n_samples)
            Confidence ratings; must be normalized to the range [0;1].
        precomputed_parameters : dict
            Provide pre-computed parameters. A dictionary with all parameters defined by the model must be passed. This
            can sometimes be useful to obtain information from the model without having to fit the model.
            [ToDO: which information?]
        guess_type2 : array-like of shape (n_params_type1)
            For testing: provide an initial guess for the optimization of the type 2 level
        verbose : bool
            If True, information of the model fitting procedure is printed.
        ignore_warnings : bool
            If True, warnings during model fitting are supressed.
        """

        # Instantiate util.Data object
        self.data = Data(self.cfg, x_stim, d_dec, c_conf)

        self.fit_type1(precomputed_parameters=precomputed_parameters, verbose=verbose, ignore_warnings=ignore_warnings)

        if not skip_type2:
            self.fit_type2(precomputed_parameters=precomputed_parameters, guess_type2=guess_type2, verbose=verbose,
                           ignore_warnings=ignore_warnings)


    def fit_type1(self, x_stim=None, d_dec=None, c_conf=None, precomputed_parameters=None, verbose=True,
                  ignore_warnings=False):

        if self.data is None:
            if x_stim is None or d_dec is None:
                raise ValueError('If the data attribute of the ReMeta instance is None, at least x_stim (stimuli) '
                                 'and d_dec (choices) have to be passed to fit_type1()')
            else:
                self.data = Data(self.cfg, x_stim, d_dec, c_conf)

        if verbose:
            print('\n+++ Type 1 level +++')
        # with warnings.catch_warnings(record=True) as w:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='scipy.optimize',
                                    message='delta_grad == 0.0. Check if the approximated function is linear. If the '
                                            'function is linear better results can be obtained by defining the Hessian '
                                            'as zero instead of using quasi-Newton approximations.')
            if ignore_warnings:
                warnings.filterwarnings('ignore')
            if isinstance(precomputed_parameters, dict):
                if not np.all([p in precomputed_parameters for p in self.cfg.paramset_type1.param_names_flat]):
                    raise ValueError('Set of precomputed type 1 parameters is incomplete.')
                self.model.fit.fit_type1 = OptimizeResult(
                    x=[precomputed_parameters[p] for p in self.cfg.paramset_type1.param_names_flat],
                    fun=self._negll_type1([precomputed_parameters[p] for p in self.cfg.paramset_type1.param_names_flat])
                )
            else:
                if self.cfg.paramset_type1.nparams > 0:
                    if verbose:
                        negll_initial_guess = self._negll_type1(self.cfg.paramset_type1.guess)
                        print(f'Initial guess (neg. LL: {negll_initial_guess:.2f})')
                        for i, p in enumerate(self.cfg.paramset_type1.param_names_flat):
                            print(f'{TAB}[guess] {p}: {self.cfg.paramset_type1.guess[i]:.4g}')
                        print('Performing local optimization')
                    t0 = timeit.default_timer()
                    self.model.fit.fit_type1 = minimize(
                        self._negll_type1, self.cfg.paramset_type1.guess, bounds=self.cfg.paramset_type1.bounds,
                        method='trust-constr'
                    )
                    if self.cfg.enable_type1_param_thresh:
                        fit_powell = minimize(
                            self._negll_type1, self.cfg.paramset_type1.guess, bounds=self.cfg.paramset_type1.bounds,
                            method='Powell'
                        )
                        if fit_powell.fun < self.model.fit.fit_type1.fun:
                            self.model.fit.fit_type1 = fit_powell
                    self.model.fit.fit_type1.execution_time = timeit.default_timer() - t0

                else:
                    self.model.fit.fit_type1 = OptimizeResult(x=None)
            if isinstance(self.cfg.true_params, dict):
                if not np.all([p in self.cfg.true_params for p in self.cfg.paramset_type1.param_names]):
                    raise ValueError('Set of provided true type 1 parameters is incomplete.')
                params_true = sum([[self.cfg.true_params[p]] if n == 1 else self.cfg.true_params[p] for n, p in
                                  zip(self.cfg.paramset_type1.param_len, self.cfg.paramset_type1.param_names)], [])
                self.model.fit.fit_type1.negll_true = self._negll_type1(params_true)

            # call once again with final=True to save the model fit
            negll_type1, params_type1, type1_likelihood, type1_posterior = \
                self._negll_type1(self.model.fit.fit_type1.x, final=True)
            if 'type1_thresh' in params_type1 and params_type1['type1_thresh'] < self.data.stimuli_min:
                warnings.warn('Fitted threshold is below the minimal stimulus intensity; consider disabling '
                              'the type 1 threshold by setting enable_type1_param_thresh to 0', category=UserWarning)
            self.model.store_type1(negll_type1=negll_type1, params_type1=params_type1, type1_likelihood=type1_likelihood,
                                   type1_posterior=type1_posterior, stimuli_max=self.data.stimuli_max)
            self.model.report_fit_type1(verbose)
            self.type1_is_fitted = True

        # if not ignore_warnings and verbose:
        #     print_warnings(w)

        self.model.params = self.model.params_type1


    def fit_type2(self, precomputed_parameters=None, guess_type2=None, verbose=True, ignore_warnings=False):

        # compute decision values
        self._compute_y_decval()

        if verbose:
            print('\n+++ Type 2 level +++')

        # args_type2 = dict(mock_binsize=None, ignore_warnings=ignore_warnings)
        args_type2 = [None, ignore_warnings]
        if precomputed_parameters is not None:
            if not np.all([p in precomputed_parameters for p in self.cfg.paramset_type2.param_names_flat]):
                raise ValueError('Set of precomputed type 2 parameters is incomplete.')
            self.model.params_type2 = {p: precomputed_parameters[p] for p in self.cfg.paramset_type2.param_names_flat}
            self.model.fit.fit_type2 = OptimizeResult(
                x=[precomputed_parameters[p] for p in self.cfg.paramset_type2.param_names_flat],
                fun=self.fun_negll_type2([precomputed_parameters[p] for p in self.cfg.paramset_type2.param_names_flat])
            )
            fitinfo_type2 = self.fun_negll_type2(list(self.model.params_type2.values()), *args_type2, final=True)  # noqa
            self.model.store_type2(**fitinfo_type2)
        else:
            with warnings.catch_warnings(record=True) as w:  # noqa
                warnings.filterwarnings('ignore', module='scipy.optimize')
                if self.cfg.paramset_type2.nparams > 0:
                    self.model.fit.fit_type2 = fmincon(
                        self.fun_negll_type2, self.cfg.paramset_type2, args_type2, gradient_free=self.cfg.gradient_free,
                        gridsearch=self.cfg.gridsearch, grid_multiproc=self.cfg.grid_multiproc,
                        global_minimization=self.cfg.global_minimization,
                        fine_gridsearch=self.cfg.fine_gridsearch,
                        gradient_method=self.cfg.gradient_method, slsqp_epsilon=self.cfg.slsqp_epsilon,
                        init_nelder_mead=self.cfg.init_nelder_mead,
                        guess=guess_type2,
                        verbose=verbose
                    )
                else:
                    self.model.fit.fit_type2 = OptimizeResult(x=None)

            # call once again with final=True to save the model fit
            fitinfo_type2 = self.fun_negll_type2(self.model.fit.fit_type2.x, *args_type2, final=True)
            # if 'type2_criteria' in fitinfo_type2['params_type2']:
            #     fitinfo_type2['params_type2']['type2_criteria'][1:] = [np.sum(fitinfo_type2['params_type2']['type2_criteria'][:i+1])
            #                                                            for i in range(1, len(fitinfo_type2['params_type2']['type2_criteria']))]
            self.model.store_type2(**fitinfo_type2)

        if self.cfg.true_params is not None:
            type2_params_true = self.cfg.true_params.copy()
            if self.cfg.enable_type2_param_criteria and not 'type2_criteria' in type2_params_true:
                type2_params_true[f'type2_criteria'] = [1/self.cfg.n_discrete_confidence_levels for _ in range(self.cfg.n_discrete_confidence_levels-1)]
            type2_params_true_values = sum([([type2_params_true[p]] if n == 1 else type2_params_true[p])
                                                        if p in type2_params_true else [None]*n for n, p in
                                            zip(self.cfg.paramset_type2.param_len, self.cfg.paramset_type2.param_names)],
                                           [])
            self.model.fit.fit_type2.negll_true = self.fun_negll_type2(type2_params_true_values)

        self.model.report_fit_type2(verbose)

        self.model.params = ({} if self.model.params is None else self.model.params) | self.model.params_type2

        self.type2_is_fitted = True

        # if not ignore_warnings:
        #     print_warnings(w)

    def summary(self, extended=False, generative=True, generative_nsamples=1000):
        """
        Provides information about the model fit.

        Parameters
        ----------
        extended : bool
            If True, store various model variables in the summary object.
        generative : bool
            If True, compare model predictions of confidence with empirical confidence by repeatedly sampling from
            the generative model.
        generative_nsamples : int
            Number of samples used for the generative model (higher = more accurate).

        Returns
        ----------
        summary : dataclass
            Information about model fit.
        """

        if self.type2_is_fitted and generative:
            c_conf_generative = simu_data(generative_nsamples, self.data.nsamples, self.model.params,
                                          cfg=self.cfg, x_stim_external=self.data.x_stim, verbose=False).c_conf
        else:
            c_conf_generative = None
        summary_model = self.model.summary(
            extended=extended, c_conf_empirical=self.data.c_conf, c_conf_generative=c_conf_generative
        )
        desc = dict(data=self.data.summary(extended), model=summary_model, cfg=self.cfg)

        summary_ = make_dataclass('Summary', desc.keys())

        def repr_(self_):
            txt = f'***{self_.__class__.__name__}***\n'
            for k, v in self_.__dict__.items():
                if k == 'cfg':
                    txt += f"\n{k}: {type(desc['cfg'])} <not displayed>"
                else:
                    txt += f"\n{k}: {v}"
            return txt

        summary_.__repr__ = repr_
        summary_.__module__ = '__main__'
        summary = summary_(**desc)
        return summary

    def _negll_type1(self, params, final=False):
        """
        Minimization function for the type 1 level

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the type 1 level.
        final : bool
            If True, store latent variables and parameters.

        Returns:
        --------
        negll: float
            Negative (summed) log likelihood.
        """

        bl = self.cfg.paramset_type1.param_len
        params_type1 = {p: params[int(np.sum(bl[:i]))] if n == 1 else [params[int(np.sum(bl[:i])) + j] for j in range(n)]
                        for i, (p, n) in enumerate(zip(self.cfg.paramset_type1.param_names, bl))}
        type1_thresh = _check_param(params_type1['type1_thresh'] if self.cfg.enable_type1_param_thresh else 0)
        type1_bias = _check_param(params_type1['type1_bias'] if self.cfg.enable_type1_param_bias else 0)

        cond_neg, cond_pos = self.data.x_stim < 0, self.data.x_stim >= 0
        y_decval = np.full(self.data.x_stim.shape, np.nan)
        y_decval[cond_neg] = (np.abs(self.data.x_stim[cond_neg]) > type1_thresh[0]) * self.data.x_stim[cond_neg] + type1_bias[0]
        y_decval[cond_pos] = (np.abs(self.data.x_stim[cond_pos]) > type1_thresh[1]) * self.data.x_stim[cond_pos] + type1_bias[1]

        if (self.cfg.type1_noise_signal_dependency != 'none') or (self.cfg.enable_type1_param_noise == 2):
            type1_noise = compute_signal_dependent_type1_noise(
                x_stim=self.data.x_stim, type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency, **params_type1
            )
        else:
            type1_noise = params_type1['type1_noise']

        posterior = logistic(y_decval, type1_noise)
        likelihood_type1 = (self.data.d_dec == 1) * posterior + (self.data.d_dec == 0) * (1 - posterior)
        negll = np.sum(-np.log(np.maximum(likelihood_type1, self.cfg.min_type1_likelihood)))

        return (negll, params_type1, likelihood_type1, posterior) if final else negll

    def _negll_type2_noisyreadout(self, params, mock_binsize=None, ignore_warnings=False, final=False):
        """
        Negative log likelihood minimization method of the noisy-readout model

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the type 2 level.
        mock_binsize : float
            Binsize around empirical confidence ratings to evaluate the likelihood. If not None, also returns the
            likelihood variable.
        ignore_warnings : bool
            If True, suppress warnings during minimization.
        final : bool
            If True, return latent variables and parameters.
            Note: this has to be the final parameter in the method definition!

        Returns:
        --------
        By default, the method returns the negative log likelihood. However, depending on the arguments various
        combinations of variables are returned (see Parameters).
        negll_type2: float
            Negative (summed) log likelihood.
        """

        params_type2, type2_likelihood = \
            self._helper_negll_type2_noisyreadout(params, mock_binsize, ignore_warnings, final)

        if final and ('type2_criteria' in params_type2) and (np.sum(params_type2['type2_criteria']) > 1.001):
            params_type2['type2_criteria'] = check_criteria_sum(params_type2['type2_criteria'])

        # compute log likelihood
        type2_cum_likelihood = np.nansum(self.model.y_decval_pmf * type2_likelihood, axis=1)
        if self.cfg.experimental_min_uniform_type2_likelihood:
            # use an upper bound for the negative log likelihood based on a uniform 'guessing' model
            negll_type2 = min(self.model.uniform_type2_negll, -np.sum(np.log(np.maximum(type2_cum_likelihood, 1e-200))))
        else:
            negll_type2 = -np.sum(np.log(np.maximum(type2_cum_likelihood, self.cfg.min_type2_likelihood)))

        if final:
            self.model.c_conf = self._type1_evidence_to_confidence(self.model.z1_type1_evidence, params_type2)
            return dict(negll_type2=negll_type2, params_type2=params_type2, type2_likelihood=type2_likelihood,
                        type2_cum_likelihood=type2_cum_likelihood,
                        z1_type1_evidence=self.model.z1_type1_evidence)
        elif mock_binsize is not None:
            return negll_type2, type2_likelihood
        else:
            return negll_type2

    def _helper_negll_type2_noisyreadout(self, params, mock_binsize=None, ignore_warnings=False, final=False):  # noqa
        """
        Helper for the negative log likelihood minimization method of the noisy-readout model

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the type 2 level.
        mock_binsize : float
            Binsize around empirical confidence ratings to evaluate the likelihood. If not None, also returns the
            likelihood variable.
        ignore_warnings : bool
            If True, suppress warnings during minimization.
        final : bool
            If True, return latent variables and parameters.
            Note: this has to be the final parameter in the method definition!

        Returns:
        --------
        params_type2, type2_likelihood
        """

        bl = self.cfg.paramset_type2.param_len
        params_type2 = {p: params[int(np.sum(bl[:i]))] if n == 1 else [params[int(np.sum(bl[:i])) + j] for j in range(n)]
                        for i, (p, n) in enumerate(zip(self.cfg.paramset_type2.param_names, bl))}


        if self.cfg.type2_fitting_type == 'criteria':
            type2_likelihood = self._compute_likelihood_noisyreadout_criteria(
                params_type2,
                criteria=params_type2['type2_criteria'] if self.cfg.enable_type2_param_criteria else None
            )
        else:
            type2_likelihood = self._compute_likelihood_noisyreadout(params_type2, mock_binsize=mock_binsize)
        if not self.cfg.experimental_include_incongruent_y_decval:
            type2_likelihood[self.model.y_decval_invalid] = np.nan

        return params_type2, type2_likelihood

    def _compute_likelihood_noisyreadout(self, params_type2, mock_binsize=None):
        binsize = self.cfg.type2_binsize if mock_binsize is None else mock_binsize
        if self.cfg.experimental_wrap_type2_integration_window:
            wrap_neg = binsize - np.abs(np.minimum(1, self.data.c_conf_2d + binsize) - self.data.c_conf_2d)  # noqa
            wrap_pos = binsize - np.abs(np.maximum(0, self.data.c_conf_2d - binsize) - self.data.c_conf_2d)  # noqa
            binsize_neg, binsize_pos = binsize + wrap_neg, binsize + wrap_pos
        else:
            binsize_neg, binsize_pos = binsize, binsize
        data_z1_type1_evidence_lb = self._confidence_to_type1_evidence(
            np.maximum(0, self.data.c_conf_2d - binsize_neg), params_type2)
        data_z1_type1_evidence_ub = self._confidence_to_type1_evidence(
            np.minimum(1, self.data.c_conf_2d + binsize_pos), params_type2)
        data_z1_type1_evidence = self._confidence_to_type1_evidence(
            self.data.c_conf_2d, params_type2)

        type2_noise_dist = get_dist(self.cfg.type2_noise_dist, mode=self.model.z1_type1_evidence, scale=params_type2['type2_noise'],
                                    logscale_min=self.cfg._type2_noise_logscale_min,
                                    type2_noise_type='noisy_readout', log_scale=self.cfg.type2_noise_logscale, experimental_lookup_table=self.lookup_table)
        if self.cfg.experimental_disable_type2_binsize:
            type2_likelihood = type2_noise_dist.pdf(data_z1_type1_evidence)
        else:
            type2_likelihood = type2_noise_dist.cdf(data_z1_type1_evidence_ub) - type2_noise_dist.cdf(data_z1_type1_evidence_lb)
        return type2_likelihood

    def _compute_likelihood_noisyreadout_criteria(self, params_type2, criteria=None):


        type2_likelihood = np.empty_like(self.model.z1_type1_evidence, float)

        if criteria is None:
            criteria_ = np.hstack((np.ones(self.cfg.n_discrete_confidence_levels - 1) / self.cfg.n_discrete_confidence_levels, 1))
        else:
            criteria_ = np.hstack((criteria, 1))

        for i, crit in enumerate(criteria_):
            cnd = self.data.c_conf_discrete == i
            if cnd.sum():
                if i == 0:
                    lower_bin_edge = 0
                    upper_bin_edge = crit
                else:
                    lower_bin_edge = np.sum(criteria_[:i])
                    upper_bin_edge = np.sum(criteria_[:i+1])
                data_z1_type1_evidence_lb = self._confidence_to_type1_evidence(lower_bin_edge, params_type2, mask=cnd)
                data_z1_type1_evidence_ub = self._confidence_to_type1_evidence(upper_bin_edge, params_type2, mask=cnd)
                type2_noise_dist = get_dist(self.cfg.type2_noise_dist, mode=self.model.z1_type1_evidence[cnd],
                                            scale=params_type2['type2_noise'], type2_noise_type='noisy_readout',
                                            log_scale=self.cfg.type2_noise_logscale,
                                            logscale_min=self.cfg._type2_noise_logscale_min,
                                            experimental_lookup_table=self.lookup_table)

                type2_likelihood[cnd] = (type2_noise_dist.cdf(data_z1_type1_evidence_ub) -
                                         type2_noise_dist.cdf(data_z1_type1_evidence_lb))

        return type2_likelihood

    def _negll_type2_noisyreport(self, params, mock_binsize=None, ignore_warnings=False, final=False):
        """
        Negative log likelihood minimization method of the noisy-report model

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the type 2 level.
        mock_binsize : float
            Binsize around empirical confidence ratings to evaluate the likelihood. If not None, also returns the
            likelihood variable.
        ignore_warnings : bool
            If True, suppress warnings during minimization.
        final : bool
            If True, return latent variables and parameters.
            Note: this has to be the final parameter in the method definition!

        Returns:
        --------
        By default, the method returns the negative log likelihood. However, depending on the arguments various
        combinations of variables are returned.
        negll_type2: float
            Negative cumulative log likelihood.
        """

        params_type2, z1_type1_evidence, type2_likelihood = \
            self._helper_negll_type2_noisyreport(params, mock_binsize, ignore_warnings, final)

        if final and ('type2_criteria' in params_type2) and (np.sum(params_type2['type2_criteria']) > 1.001):
            params_type2['type2_criteria'] = check_criteria_sum(params_type2['type2_criteria'])

        # compute weighted cumulative negative log likelihood
        type2_cum_likelihood = np.nansum(type2_likelihood * self.model.y_decval_pmf, axis=1)
        if self.cfg.experimental_min_uniform_type2_likelihood:
            # use an upper bound for the negative log likelihood based on a uniform 'guessing' model
            negll_type2 = min(self.model.uniform_type2_negll, -np.sum(np.log(np.maximum(type2_cum_likelihood, 1e-200))))
        else:
            negll_type2 = -np.sum(np.log(np.maximum(type2_cum_likelihood, self.cfg.min_type2_likelihood)))

        if final:
            return dict(negll_type2=negll_type2, params_type2=params_type2, type2_likelihood=type2_likelihood,
                        z1_type1_evidence=z1_type1_evidence, type2_cum_likelihood=type2_cum_likelihood)
        elif mock_binsize is not None:
            return negll_type2, type2_likelihood
        else:
            return negll_type2

    def _helper_negll_type2_noisyreport(self, params, mock_binsize=None, ignore_warnings=False, final=False):
        """
        Helper for the negative log likelihood minimization method of the noisy-report model

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the type 2 level.
        mock_binsize : float
            Binsize around empirical confidence ratings to evaluate the likelihood. If not None, also returns the
            likelihood variable.
        ignore_warnings : bool
            If True, suppress warnings during minimization.
        final : bool
            If True, return latent variables and parameters.
            Note: this has to be the final parameter in the method definition!

        Returns:
        --------
        params_type2, z1_type1_evidence, type2_likelihood
        """

        bl = self.cfg.paramset_type2.param_len
        params_type2 = {p: params[int(np.sum(bl[:i]))] if n == 1 else [params[int(np.sum(bl[:i])) + j] for j in range(n)]
                       for i, (p, n) in enumerate(zip(self.cfg.paramset_type2.param_names, bl))}

        if hasattr(self.data, 'z1_type1_evidence'):
            z1_type1_evidence = self.data.z1_type1_evidence
        else:
            z1_type1_evidence = self.model.z1_type1_evidence

        self.model.c_conf = self._type1_evidence_to_confidence(z1_type1_evidence, params_type2)

        if self.cfg.type2_fitting_type == 'criteria':
            type2_likelihood = self._compute_likelihood_noisyreport_criteria(
                params_type2['type2_noise'], params_type2['type2_criteria'] if self.cfg.enable_type2_param_criteria else None
            )
        else:
            type2_likelihood = self._compute_likelihood_noisyreport(params_type2['type2_noise'], mock_binsize=mock_binsize)

        if not self.cfg.experimental_include_incongruent_y_decval:
            type2_likelihood[self.model.y_decval_invalid] = np.nan
            self.model.c_conf[self.model.y_decval_invalid] = np.nan

        return params_type2, z1_type1_evidence, type2_likelihood


    def _compute_likelihood_noisyreport_criteria(self, type2_noise, criteria=None):

        # criteria = [0, 0.33, 0.4, 1]
        # criteria = [0.2, 0.4, 0.6, 0.8]
        # criteria = [0.29999645, 0.49090545, 0.68181526, 0.87272674]

        type2_likelihood = np.empty_like(self.model.c_conf, float)
        # type2_likelihood_unnorm = np.empty_like(self.model.c_conf, float)
        # nsamples = 0

        if criteria is None:
            criteria_ = np.hstack((np.ones(self.cfg.n_discrete_confidence_levels - 1) / self.cfg.n_discrete_confidence_levels, 1))
        else:
            criteria_ = np.hstack((criteria, 1))

        # nsamples = 0
        for i, crit in enumerate(criteria_):
            cnd = self.data.c_conf_discrete == i
            if cnd.sum():
                if i == 0:
                    lower_bin_edge = 0
                    upper_bin_edge = crit
                else:
                    lower_bin_edge = np.sum(criteria_[:i])
                    upper_bin_edge = np.sum(criteria_[:i+1])
                # nsamples += cnd.sum()
                dist = get_dist(self.cfg.type2_noise_dist, mode=self.model.c_conf[cnd], scale=type2_noise,
                                type2_noise_type='noisy_report', log_scale=self.cfg.type2_noise_logscale,
                                logscale_min=self.cfg._type2_noise_logscale_min,
                                experimental_lookup_table=self.lookup_table)
                type2_likelihood[cnd] = (dist.cdf(min(1, upper_bin_edge)) - dist.cdf(lower_bin_edge))
                # print(i, lower_bin_edge, upper_bin_edge, -np.log(type2_likelihood[cnd, 50]).sum(), cnd.sum())

        # print(f'{nsamples=}')
        # print(-np.log(type2_likelihood[:, 50]).sum())

        return type2_likelihood


    def _compute_likelihood_noisyreport(self, type2_noise, mock_binsize=None):
        binsize = self.cfg.type2_binsize if mock_binsize is None else mock_binsize
        if self.cfg.experimental_wrap_type2_integration_window:
            wrap_neg = binsize - np.abs(np.minimum(1, self.data.c_conf_2d + binsize) - self.data.c_conf_2d)
            wrap_pos = binsize - np.abs(np.maximum(0, self.data.c_conf_2d - binsize) - self.data.c_conf_2d)
            binsize_neg, binsize_pos = binsize + wrap_neg, binsize + wrap_pos
        else:
            binsize_neg, binsize_pos = binsize, binsize
        dist = get_dist(self.cfg.type2_noise_dist, mode=self.model.c_conf, scale=type2_noise,
                        type2_noise_type='noisy_report', log_scale=self.cfg.type2_noise_logscale,
                        logscale_min=self.cfg._type2_noise_logscale_min,
                        experimental_lookup_table=self.lookup_table)
        # compute the probability of the actual confidence ratings given the pred confidence
        if self.cfg.experimental_disable_type2_binsize:
            type2_likelihood = dist.pdf(self.data.c_conf_2d)
        else:
            with warnings.catch_warnings():
                # catch this warning, which doesn't make any sense (output is valid if this happens)
                warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.stats',
                                        message='divide by zero encountered in _beta_cdf')
                type2_likelihood = dist.cdf(np.minimum(1, self.data.c_conf_2d + binsize_pos)) - \
                                   dist.cdf(np.maximum(0, self.data.c_conf_2d - binsize_neg))  # noqa

        return type2_likelihood

    def _type1_evidence_to_confidence(self, z1_type1_evidence, params_type2):
        """
        Helper function to convert type 1 evidence to confidence
        """
        return type1_evidence_to_confidence(
            z1_type1_evidence=z1_type1_evidence, y_decval=self.model.y_decval,
            x_stim=self.data.x_stim, type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency,
            **self.model.params_type1, **params_type2
        )

    def _confidence_to_type1_evidence(self, c_conf, params_type2, mask=None):
        """
        Helper function to concert confidence to type 1 evidence
        """
        if mask is None:
            return confidence_to_type1_evidence(
                c_conf=c_conf, x_stim=self.data.x_stim,
                y_decval=self.model.y_decval, type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency,
                **self.model.params_type1, **params_type2
            )
        else:
            return confidence_to_type1_evidence(
                c_conf=c_conf, x_stim=self.data.x_stim[mask],
                y_decval=self.model.y_decval[mask], type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency,
                **self.model.params_type1, **params_type2
            )

    def _compute_y_decval(self):
        """
        Compute type 1 decision values
        """

        type1_noise_trialwise = compute_signal_dependent_type1_noise(
            x_stim=self.data.x_stim_2d, type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency, **self.model.params_type1
        )
        type1_thresh = _check_param(self.model.params_type1['type1_thresh'] if self.cfg.enable_type1_param_thresh else 0)
        type1_bias = _check_param(self.model.params_type1['type1_bias'] if self.cfg.enable_type1_param_bias else 0)

        cond_neg, cond_pos = (self.data.x_stim_2d < 0).squeeze(), (self.data.x_stim_2d >= 0).squeeze()
        y_decval_mode = np.full(self.data.x_stim_2d.shape, np.nan)
        y_decval_mode[cond_neg] = (np.abs(self.data.x_stim_2d[cond_neg]) >= type1_thresh[0]) * \
            self.data.x_stim_2d[cond_neg] + type1_bias[0]
        y_decval_mode[cond_pos] = (np.abs(self.data.x_stim_2d[cond_pos]) >= type1_thresh[1]) * \
            self.data.x_stim_2d[cond_pos] + type1_bias[1]

        range_ = np.linspace(0, self.cfg.y_decval_range_nsds, int((self.cfg.y_decval_range_nbins + 1) / 2))[1:]
        yrange = np.hstack((-range_[::-1], 0, range_))
        self.model.y_decval = np.full((y_decval_mode.shape[0], yrange.shape[0]), np.nan)
        self.model.y_decval[cond_neg] = y_decval_mode[cond_neg] + yrange * np.mean(type1_noise_trialwise[cond_neg])
        self.model.y_decval[cond_pos] = y_decval_mode[cond_pos] + yrange * np.mean(type1_noise_trialwise[cond_pos])

        logistic_neg = logistic_dist(loc=y_decval_mode[cond_neg], scale=type1_noise_trialwise[cond_neg] * np.sqrt(3) / np.pi)
        logistic_pos = logistic_dist(loc=y_decval_mode[cond_pos], scale=type1_noise_trialwise[cond_pos] * np.sqrt(3) / np.pi)
        margin_neg = type1_noise_trialwise[cond_neg] * self.cfg.y_decval_range_nsds / self.cfg.y_decval_range_nbins
        margin_pos = type1_noise_trialwise[cond_pos] * self.cfg.y_decval_range_nsds / self.cfg.y_decval_range_nbins
        self.model.y_decval_pmf = np.full(self.model.y_decval.shape, np.nan)
        self.model.y_decval_pmf[cond_neg] = (logistic_neg.cdf(self.model.y_decval[cond_neg] + margin_neg) -
                                             logistic_neg.cdf(self.model.y_decval[cond_neg] - margin_neg))
        self.model.y_decval_pmf[cond_pos] = (logistic_pos.cdf(self.model.y_decval[cond_pos] + margin_pos) -
                                             logistic_pos.cdf(self.model.y_decval[cond_pos] - margin_pos))
        # normalize PMF
        self.model.y_decval_pmf = self.model.y_decval_pmf / self.model.y_decval_pmf.sum(axis=1).reshape(-1, 1)
        # invalidate invalid decision values
        if not self.cfg.experimental_include_incongruent_y_decval:
            self.model.y_decval_invalid = np.sign(self.model.y_decval) != \
                                          np.sign(self.data.d_dec_2d - 0.5)
            self.model.y_decval_pmf[self.model.y_decval_invalid] = np.nan

        if self.cfg.experimental_min_uniform_type2_likelihood:
            # self.cfg.type2_binsize*2 is the probability for a given confidence rating assuming a uniform
            # distribution for confidence. This 'confidence guessing model' serves as a upper bound for the
            # negative log likelihood.
            min_type2_likelihood = self.cfg.type2_binsize * 2 * np.ones(self.model.y_decval_pmf.shape)
            min_cum_type2_likelihood = np.nansum(min_type2_likelihood * self.model.y_decval_pmf, axis=1)
            self.model.uniform_type2_negll = -np.log(min_cum_type2_likelihood).sum()

        self.model.z1_type1_evidence = np.abs(self.model.y_decval)
        self.model.y_decval_mode = y_decval_mode.flatten()


    def _check_fit(self):
        if not self.type1_is_fitted and not self.type2_is_fitted:
            raise RuntimeError('Please fit the model before plotting.')
        elif self.type1_is_fitted and not self.type2_is_fitted:
            raise RuntimeError('Only the type 1 level was fitted. Please also fit the type 2 level to plot'
                               'a link function.')

    def plot_evidence_versus_confidence(self, **kwargs):
        self._check_fit()
        plot_evidence_versus_confidence(
            self.data.x_stim, self.data.c_conf, self.model.y_decval_mode, self.model.params, cfg=self.cfg, **kwargs
        )

    def plot_confidence_dist(self, **kwargs):
        self._check_fit()
        varlik = self.model.z1_type1_evidence if self.cfg.type2_noise_type == 'noisy_readout' else self.model.c_conf
        plot_confidence_dist(
            self.cfg, self.data.x_stim, self.data.c_conf, self.model.params, var_likelihood=varlik,
            y_decval=self.model.y_decval,
            likelihood_weighting=self.model.y_decval_pmf, **kwargs
        )


def load_dataset(name, verbose=True, return_params=False, return_y_decval=False, return_cfg=False):
    import gzip
    path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'demo_data', f'example_data_{name}.pkl.gz')
    if os.path.exists(path):
        with gzip.open(path, 'rb') as f:
            stimuli, choices, confidence, params, cfg, y_decval, stats = pickle.load(f)
    else:
        raise FileNotFoundError(f'[Dataset does not exist!] No such file: {path}')

    if verbose:
        print(f"Loading dataset '{name}' which was generated as follows:")
        print('..Generative model:')
        print(f'{TAB}Type 2 noise type: {cfg.type2_noise_type}')
        print(f'{TAB}Type 2 noise distribution: {cfg.type2_noise_dist}')
        print('..Generative parameters:')
        for i, (k, v) in enumerate(params.items()):
            if k == 'type2_criteria':
                type2_criteria_absolute = [np.sum(params['type2_criteria'][:j+1]) for j in range(len(params['type2_criteria']))]
                print(f"{TAB}{k}: [{', '.join([f'{x:.4g}' for x in v])}] = gaps | criteria = [{', '.join([f'{x:.4g}' for x in type2_criteria_absolute])}]")
            else:
                if hasattr(v, '__len__'):
                    print(f"{TAB}{k}: [{', '.join([f'{x:.4g}' for x in v])}]")
                else:
                    print(f'{TAB}{k}: {v:.4g}')
        print('..Characteristics:')
        print(f'{TAB}No. subjects: {1 if stimuli.ndim == 1 else len(stimuli)}')
        print(f'{TAB}No. samples: {stimuli.shape[0] if stimuli.ndim == 1 else stimuli.shape[1]}')
        print(f"{TAB}Type 1 performance: {100*stats['performance']:.1f}%")
        if 'confidence' in stats:
            print(f"{TAB}Avg. confidence: {stats['confidence']:.3f}")
        if 'mratio' in stats:
            print(f"{TAB}M-Ratio: {stats['mratio']:.3f}")
        if 'type2_criteria' in params:
            type2_criteria_bias = np.mean(params['type2_criteria'])*(len(params['type2_criteria'])+1)-1
            print(f'{TAB}Criterion bias: {type2_criteria_bias:.5g}')

    return_list = [stimuli, choices, confidence]
    if return_params:
        return_list += [params]
    if return_cfg:
        return_list += [cfg]
    if return_y_decval:
        return_list += [y_decval]
    return tuple(return_list)
