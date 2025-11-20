import warnings
from dataclasses import dataclass
from typing import Dict, Union, List

import numpy as np
from importlib.util import find_spec

try:
    from .modelspec import Parameter, ParameterSet
    from .util import ReprMixin, _slsqp_epsilon
except ImportError:
    from remeta_v1.remeta.modelspec import Parameter, ParameterSet
    from remeta_v1.remeta.util import ReprMixin, _slsqp_epsilon


@dataclass
class Configuration(ReprMixin):
    """
    Configuration for the ReMeta toolbox

    Parameters
    ----------
    *** Basic definition of the model ***
    type2_fitting_type : str (default: 'criteria')
        Whether confidence is fitted with discrete *criteria* or as a continuous variable.
        Possible values: 'criteria', 'continuous'
    n_discrete_confidence_levels : int (default: 5)
        Number of confidence criteria. Only applies in case of type2_fitting_type='criteria'.
    type2_noise_type : str (default: 'noisy-report)
        Whether the model considers noise at readout or report.
        Possible values: 'noisy_report', 'noisy_readout'
    type2_noise_dist : str
            (default: noisy-report + criteria -> 'beta'
                      noisy-report + continuous -> 'truncated_norm'
                      noisy-readout + criteria -> 'gamma'
                      noisy-readout + continuous -> 'truncated_norm'
            )
        Metacognitive noise distribution.
        Possible values:
            'truncated_norm', 'truncated_gumbel', 'truncated_lognorm'
            noisy-report only: 'beta'
            noisy-readout only: 'lognorm', 'gamma'
         Experimental (for internal use only):
             'truncated_norm_fit', 'truncated_lognorm_varstd',
             noisy-report only: 'beta_std'
             noisy-readout only: 'betaprime', 'lognorm_varstd',
         Works only with lookup table (not deployed via pip):
            'truncated_norm_lookup', 'truncated_gumbel_lookup'



    *** Enable or disable specific parameters ***
    * Each setting can take the values 0, 1 or 2:
    *    0: Disable parameter.
    *    1: Enable parameter.
    *    2: Enable parameter and fit separate values for the negative and positive stimulus category
            (works only for type 1 parameters!)
    enable_type1_param_noise : int (default: 1)
        Fit separate type 1 noise parameters for both stimulus categories.
    enable_type1_param_noise_heteroscedastic : int (default: 0)
        Fit an additional type 1 noise parameter for signal-dependent type 1 noise (the type of dependency is
        defined via `type1_noise_signal_dependency`).
    enable_type1_param_thresh : int (default: 0)
        Fit a type 1 threshold.
    enable_type1_param_bias : int (default: 1)
        Fit a type 1 bias towards one of the stimulus categories.
    enable_type2_param_noise : int (default: 1)
        Fit a metacognitive noise parameter
    enable_type2_param_evidence_bias_mult : int (default: 0)
        Fit a multiplicative metacognitive bias loading on evidence.
    enable_type2_param_criteria : int (default: 0)
        Fit confidence criteria.

    *** Define fitting characteristics of the parameters ***
    * The fitting of each parameter is characzerized as follows:
    *     1) An initial guess.
    *     2) Lower and upper bound.
    *     3) Grid linspace, i.e. the range of values that are tested during the initial gridsearch search in
    *        "numpy linspace" format (lower, upper, number_of_grid_points).
    * Sensible default values are provided for all parameters. To tweak those, one can either define an entire
    * ParameterSet, which is a container for a set of parameters, or each parameter individually. Note that the
    * parameters must be either defined as a Parameter instance or as List[Parameter] in case when separate values are
    * fitted for the positive and negative stimulus category/decision value).
    paramset_type1 : ParameterSet
        Parameter set for the type 1 level.
    paramset_type2 : ParameterSet
        Parameter set for the type 2 level.

    type1_param_noise : Union[Parameter, List[Parameter]]  (default: 1)
        Parameter for type 1 noise.
    type1_param_noise_heteroscedastic : Union[Parameter, List[Parameter]]  (default: 0)
        Parameter for signal-dependent type 1 noise.
    type1_noise_signal_dependency: str (default: 'none')
        Can be one of 'none', 'multiplicative', 'power', 'exponential' or 'logarithm'.
    type1_param_thresh : Union[Parameter, List[Parameter]] (default: 0)
        Parameter for the type 1 threshold.
    type1_param_bias : Union[Parameter, List[Parameter]]  (default: 1)
        Parameter for the type 1 bias.
    type2_param_noise : Union[Parameter, List[Parameter]]  (default: 1)
        Parameter for metacognitive noise.
    type2_param_evidence_bias_mult : Union[Parameter, List[Parameter]]  (default: 0)
        Parameter for a multiplicative metacognitive bias loading on evidence.
    type2_param_confidence_criteria : List[Parameter]  (default: 1)
        List of parameter specifying the confidence criteria.

    *** Methodoligcal aspects of parameter fitting ***
    * Note: this applies to the fitting of type 2 parameters only.
    gridsearch : bool (default: False)
        If True, perform initial (usually coarse) gridsearch search, based on the gridsearch defined for a Parameter.
    fine_gridsearch : bool (default: False)
        If True, perform an iteratively finer gridsearch search for each parameter.
    grid_multiproc : bool (default: False)
        If True, use all available cores for the gridsearch search. If False, use a single core.
    global_minimization : bool (default: False)
        If True, use a global minimization routine.
    gradient_method : str or Tuple/List (default: 'slsqp')
        Set scipy.optimize.minimize gradient method
        If provided as Tuple/List, test different gradient methods and take the best
    gradient_free : bool (default: False)
        If True, use a gradien-free optimization routine.
    slsqp_epsilon : float or Tuple/List (default: None)
        Set parameter epsilon parameter for the SLSQP optimization method.
        If provided as Tuple/List, test different eps parameters and take the best
    init_nelder_mead : bool (default: False)
        If True, jump-start parameter minimization with the gradient-free Nelder-Mead algorithm


    *** Preprocessing ***
    normalize_stimuli_by_max : bool (default: True)
        If True, normalize provided stimuli by their maximum value.

    *** Parameters for the type 2 likelihood computation ***
    min_type1_likelihood : float
        Minimum probability used during the type 1 likelihood computation
    min_type2_likelihood : float
        Minimum probability used during the type 2 likelihood computation
    type2_binsize : float
        Integration bin size for the computation of the likelihood around empirical confidence values (noisy-report)
        or metacognitive evidence (noisy-readout).
    y_decval_range_nsds : int
        Number of standard deviations around the mean considered for type 1 uncertainty.
    y_decval_range_nbins : int
        Number of discrete decision values bins that are considered to represent type 1 uncertainty.
    experimental_min_uniform_type2_likelihood : bool
        Instead of using a minimum probability during the likelihood computation, use a maximum cumulative
        likelihood based on a 'guessing' model
    experimental_wrap_type2_integration_window : bool (default: False)
        Ensure constant window size for likelihood integration at the bounds.
        Only applies in case of type2_fitting_type='continuous' and experimental_disable_type2_binsize=False
    experimental_include_incongruent_y_decval : bool (default: False)
        Include incongruent decision values (i.e., sign(actual choice) != sign(decision value)) for the likelihood
        computation
    experimental_disable_type2_binsize : bool (default: None)
        Do not use an integegration window for likelihood computation.
        Only applies in case of type2_fitting_type='continuous'


    *** Other ***
    true_params : Dict
        Pass true (known) parameter values. This can be useful for testing to compare the likelihood of true and
        fitted parameters. The likelihood of true parameters is returned (and printed).
    initilialize_fitting_at_true_params : bool (default: False)
        Option to initialize the parameter fitting procedure at the true parameters; this can be helpful for testing.
    force_settings : bool
        Some setting combinations are known to be incompatible and/or to produce biased fits. If True, fit the model
        nevertheless.
    settings_ignore_warnings : bool (default: False)
        If True, ignore warnings about user-specified settings.
    print_configuration : bool (default: True)
        If True, print the configuration at instatiation of the ReMeta class.
    """

    type2_fitting_type: str = 'criteria'
    n_discrete_confidence_levels: int = 5
    type2_noise_type: str = 'noisy_report'
    type2_noise_dist: str = None
        # noisy-report + criteria -> 'beta'
        # noisy-report + continuous -> 'truncated_norm'
        # noisy-readout + criteria -> 'gamma'
        # noisy-readout + continuous -> 'truncated_norm'

    enable_type1_param_noise: int = 1
    enable_type1_param_thresh: int = 0
    enable_type1_param_bias: int = 1
    enable_type2_param_noise: int = 1
    enable_type2_param_evidence_bias_mult: int = 0
    enable_type2_param_criteria : int = 1
    # Experimental:
    enable_type1_param_noise_heteroscedastic: int = 0

    paramset_type1: ParameterSet = None
    paramset_type2: ParameterSet = None

    type1_param_noise: Union[Parameter, List[Parameter]] = None
    type1_param_noise_heteroscedastic: Union[Parameter, List[Parameter]] = None
    type1_param_thresh: Union[Parameter, List[Parameter]] = None
    type1_param_bias: Union[Parameter, List[Parameter]] = None
    type2_param_noise: Union[Parameter, List[Parameter]] = None
    type2_param_evidence_bias_mult: Union[Parameter, List[Parameter]] = None
    type2_param_criteria: List[Parameter] = None

    type1_noise_signal_dependency: str = 'none'

    gridsearch: bool = False
    fine_gridsearch: bool = False
    grid_multiproc: bool = False
    global_minimization: bool = False
    gradient_method: str = 'slsqp'
    gradient_free: bool = False
    slsqp_epsilon: float = None
    init_nelder_mead: bool = False

    normalize_stimuli_by_max: bool = True
    confidence_bounds_error: float = 0

    min_type2_likelihood: float = 1e-10
    min_type1_likelihood: float = 1e-10
    type2_binsize: float = 0.01
    y_decval_range_nsds: int = 5
    y_decval_range_nbins: int = 101
    experimental_min_uniform_type2_likelihood: bool = False
    experimental_wrap_type2_integration_window: bool = False
    experimental_include_incongruent_y_decval: bool = False
    experimental_disable_type2_binsize: bool = False
    experimental_discrete_type2_fitting: bool = False

    true_params: Dict = None
    initilialize_fitting_at_true_params: bool = False
    force_settings: bool = False
    settings_ignore_warnings: bool = False
    print_configuration: bool = False

    type2_param_noise_min: float = 0.001

    _type1_param_noise_heteroscedastic_default: Parameter = Parameter(guess=0, bounds=(0, 10), grid_linspace=(0, 1, 5))
    _type1_param_noise_default: Parameter = Parameter(guess=0.5, bounds=(1e-3, 100), grid_linspace=(0.1, 1, 8))
    _type1_param_thresh_default: Parameter = Parameter(guess=0, bounds=(0, 1), grid_linspace=(0, 0.2, 5))
    _type1_param_bias_default: Parameter = Parameter(guess=0, bounds=(-1, 1), grid_linspace=(-0.2, 0.2, 8))
    _type2_param_noise_default: Parameter = Parameter(guess=0.2, bounds=(1e-2, 1), grid_linspace=(0.05, 1, 8))
    _type2_param_evidence_bias_mult_default: Parameter = Parameter(guess=1, bounds=(0.5, 2),
                                                                   grid_linspace=(0.5, 2, 8))

    def setup(self):

        if self.slsqp_epsilon is None:
            if self.type2_noise_type == 'noisy_readout':
                self.slsqp_epsilon = 1e-4
            else:
                self.slsqp_epsilon = _slsqp_epsilon

        if find_spec('multiprocessing_on_dill') is None:
            warnings.warn(f'Multiprocessing on dill is not installed. Setting grid_multiproc is changed to False.')
            self.grid_multiproc = False

        if self.type2_noise_type == 'noisy_report':
            if self.type2_fitting_type == 'criteria':
                self.type2_noise_dist = 'beta'
            else:
                self.type2_noise_dist = 'truncated_norm'
        else:
            if self.type2_fitting_type == 'criteria':
                self.type2_noise_dist = 'gamma'
            else:
                self.type2_noise_dist = 'truncated_norm'

        self._check_compatibility()

        self._prepare_params_type1()
        self._prepare_params_type2()
        if self.print_configuration:
            self.print()

    def _check_compatibility(self):

        if not self.settings_ignore_warnings:

            if not self.enable_type2_param_noise:
                warnings.warn(f'Setting enable_type2_param_noise=False was provided -> type2_param_noise is set to its default value '
                              f'({self.type2_param_noise_default}). You may change this value via the configuration.')

            if self.enable_type2_param_criteria and self.enable_type2_param_evidence_bias_mult:
                warnings.warn(
                    'enable_type2_param_criteria=True in combination with enable_type2_param_evidence_bias_mult=True\n'
                    'can lead to biased parameter inferences. Use with caution.')

    def _prepare_params_type1(self):
        if self.paramset_type1 is None:

            param_names_type1 = []
            params_type1 = ('noise', 'noise_heteroscedastic', 'thresh', 'bias')
            for param in params_type1:
                if getattr(self, f'enable_type1_param_{param}'):
                    param_names_type1 += [f'type1_{param}']
                    if getattr(self, f'type1_param_{param}') is None:
                        param_default = getattr(self, f'_type1_param_{param}_default')
                        if getattr(self, f'enable_type1_param_{param}') == 2:
                            setattr(self, f'type1_param_{param}', [param_default, param_default])
                        else:
                            setattr(self, f'type1_param_{param}', param_default)
                        if self.true_params is not None and self.initilialize_fitting_at_true_params and f'type2_{param}' in self.true_params:
                            getattr(self, f'type1_param_{param}').guess = self.true_params[f'type1_{param}']

            parameters = {k: getattr(self, f"type1_param_{k.split('type1_')[1]}") for k in param_names_type1}
            self.paramset_type1 = ParameterSet(parameters, param_names_type1)

    def _prepare_params_type2(self):

        if self.paramset_type2 is None:

            if self.enable_type2_param_noise and self.type2_param_noise is None:
                if self.type2_noise_dist == 'beta':
                    self._type2_param_noise_default.bounds = (1e-5, 0.5)
                    self._type2_param_noise_default.grid_linspace = (0.05, 0.45, 10)
                elif self.type2_noise_type == 'noisy_readout':
                    self._type2_param_noise_default.bounds = (1e-5, 250)

            param_names_type2 = []
            params_type2 = ('noise', 'evidence_bias_mult')
            for param in params_type2:
                if getattr(self, f'enable_type2_param_{param}'):
                    param_names_type2 += [f'type2_{param}']
                    if getattr(self, f'type2_param_{param}') is None:
                        param_default = getattr(self, f'_type2_param_{param}_default')
                        if getattr(self, f'enable_type2_param_{param}') == 2:
                            setattr(self, f'type2_param_{param}', [param_default.copy(), param_default.copy()])
                        else:
                            setattr(self, f'type2_param_{param}', param_default.copy())
                        if self.true_params is not None and self.initilialize_fitting_at_true_params and f'type2_{param}' in self.true_params:
                            getattr(self, f'type2_param_{param}').guess = self.true_params[f'type2_{param}']


            if self.enable_type2_param_criteria:
                guess_criteria = np.linspace(1 / self.n_discrete_confidence_levels, 1 - 1 / self.n_discrete_confidence_levels, self.n_discrete_confidence_levels - 1)
                grid_window = 1 / self.n_discrete_confidence_levels
                param_names_type2 += [f'type2_criteria']
                setattr(self, f'type2_param_criteria',
                        [Parameter(
                                   # guess=guess,
                                   guess=self.true_params['type2_criteria'][i] if
                                        self.true_params is not None and self.initilialize_fitting_at_true_params and
                                        'type2_criteria' in self.true_params else 1 / self.n_discrete_confidence_levels,
                                   bounds=(0, 1),
                                   # grid_linspace=(max(0, guess - grid_window), min(1, guess + grid_window), 4))
                                   grid_linspace=(0.05, 2 / self.n_discrete_confidence_levels, 4))
                         for i, guess in enumerate(guess_criteria)]
                        )

            parameters = {k: getattr(self, f"type2_param_{k.split('type2_')[1]}") for k in param_names_type2}
            self.paramset_type2 = ParameterSet(parameters, param_names_type2)
        self.check_type2_constraints()

    def print(self):
        # print('***********************')
        print(f'{self.__class__.__name__}')
        for k, v in self.__dict__.items():
            # if not self.skip_type2 or ('type2' not in k):
            print('\n'.join([f'\t{k}: {v}']))
        # print('***********************')

    def __repr__(self):
        txt = f'{self.__class__.__name__}\n'
        txt += '\n'.join([f'\t{k}: {v}' for k, v in self.__dict__.items()])
        return txt

    def check_type2_constraints(self):
        pass
        # if self.enable_type2_param_criteria:
        #     from scipy.optimize import NonlinearConstraint
        #
        #     def crit_order_fun_ineq(theta):
        #         crit = theta[-self.n_discrete_confidence_levels + 1:]
        #         return np.sum([-int(crit[i] <= (-1e-8 if i == 0 else crit[i - 1])) for i in range(len(crit))])
        #
        #     def crit_order_fun(theta):
        #         return np.diff(theta[-self.n_discrete_confidence_levels + 1:])  # [k2 - k1, k3 - k2, ...]
        #
        #     eps = 1e-4  # minimum spacing between criteria
        #     self.paramset_type2.constraints = [dict(
        #         type='ineq',
        #         fun=crit_order_fun_ineq,
        #         constraint=NonlinearConstraint(
        #             fun=crit_order_fun,
        #             lb=np.full(self.n_discrete_confidence_levels - 2, eps),  # diff >= eps
        #             ub=np.full(self.n_discrete_confidence_levels - 2, np.inf)
        #         )
        #     )]
