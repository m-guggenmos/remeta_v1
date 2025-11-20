Go directly to:
- [**Installation**](https://github.com/m-guggenmos/remeta_v1/blob/main/INSTALL.md)
- [**Basic Usage**](https://github.com/m-guggenmos/remeta_v1/blob/main/demo/basic_usage.ipynb)
- [**Common use cases**](https://github.com/m-guggenmos/remeta_v1/blob/main/demo/common_use_cases.ipynb)


# ReMeta Toolbox

The ReMeta toolbox allows researchers to estimate latent type 1 and type 2 parameters based on data of cognitive or perceptual decision-making tasks with two response categories. 


### Minimal example
Three types of data are required to fit a model:

<!---  Table --->
| Type       | Variable |Description
|------------|----------|----------|
| Stimuli    | `x_stim`   | list/array of signed stimulus intensity values, where the sign codes the stimulus category and the absolute value codes the intensity. The stimuli should be normalized to [-1; 1], although there is a setting (`normalize_stimuli_by_max`) to auto-normalize stimuli         |
| Choices    | `d_dec`    | list/array of choices coded as 0 (or alternatively -1) for the negative stimuli category and 1 for the positive stimulus category.         |
| Confidence | `c_conf`   | list/array of confidence ratings. Confidence ratings must be normalized to [0; 1]. Discrete confidence ratings must be normalized accordingly (e.g., if confidence ratings are 1-4, subtract 1 and divide by 3).         |

A minimal example would be the following:
```python
# Minimal example
import remeta
x_stim, d_dec, c_conf = remeta.load_dataset('simple')  # load example dataset
rem = remeta.ReMeta()
rem.fit(x_stim, d_dec, c_conf)
```
Output:
```
Loading dataset 'simple' which was generated as follows:
..Generative model:
    Type 2 noise type: noisy_report
    Type 2 noise distribution: truncated_norm
    Confidence link function: bayesian_confidence
..Generative parameters:
    type1_noise: 0.7
    type1_bias: 0.2
    type2_noise: 0.1
    type2_evidence_bias_mult: 1.2
..Characteristics:
    No. subjects: 1
    No. samples: 1000
    Type 1 performance: 78.5%
    Avg. confidence: 0.668
    M-Ratio: 0.921
+++ Type 1 level +++
Initial guess (neg. LL: 1902.65)
    [guess] type1_noise: 0.1
    [guess] type1_bias: 0
Performing local optimization
    [final] type1_noise: 0.745
    [final] type1_bias: 0.24
Final neg. LL: 461.45
Total fitting time: 0.13 secs
+++ Type 2 level +++
Initial guess (neg. LL: 1938.81)
    [guess] type2_noise: 0.2
    [guess] type2_evidence_bias_mult: 1
Grid search activated (grid size = 160)
    [grid] type2_noise: 0.15
    [grid] type2_evidence_bias_mult: 1.3
Grid neg. LL: 1879.1
Grid runtime: 3.22 secs
Performing local optimization
    [final] type2_noise: 0.102
    [final] type2_evidence_bias_mult: 1.21
Final neg. LL: 1872.24
Total fitting time: 3.9 secs
```

Since the dataset is based on simulation, we know the true parameters (in brackets above) of the underlying generative model, which are indeed quite close to the fitted parameters.

We can access the fitted parameters by invoking the `summary()` method on the `ReMeta` instance:

```python
# Access fitted parameters
result = rem.summary()
for k, v in result.model.params.items():
    print(f'{k}: {v:.3f}')
```

Ouput:
```
type1_noise: 0.745
type1_bias: 0.240
type2_noise: 0.102
type2_evidence_bias_mult: 1.213
```

By default, the model fits parameters for type 1 noise (`type1_noise`) and a type 1 bias (`type1_bias`), as well as metacognitive 'type 2' noise (`type2_noise`) and a metacognitive bias (`type2_evidence_bias_mult`). Moreover, by default the model assumes that metacognitive noise occurs at the stage of the confidence report (setting `type2_noise_type='noisy_report'`), that observers aim at reporting Bayesian confidence (setting `confidence_link_function='bayesian_confidence'`) and that type 2 metacognitive noise can be described by a truncated normal distribution (setting `type2_noise_dist='truncated_norm'`).

All settings can be changed via the `Configuration` object which is optionally passed to the `ReMeta` instance. For example:

```python
cfg = remeta.Configuration()
cfg.type2_noise_type = 'noisy_readout'
rem = remeta.ReMeta(cfg)
...
```

### Supported parameters

_Type 1 parameters_:
- `type1_noise`: type 1 noise
- `type1_bias`: type 1 bias towards one of the two stimulus categories
- `type1_thresh`: a (sensory) threshold, building on the assumption that a certain minimal stimulus intensity is required to elicit behavior; use only if there are stimulus intensities close to threshold
- `type1_noise_heteroscedastic`: parameter to specify stimulus-dependent type 1 noise (e.g. multiplicative noise)

_Type 2 (metacognitive) parameters:_
- `type2_noise`: metacognitive noise
- `type2_evidence_bias_mult`: multiplicative metacognitive bias
- `type2_evidence_bias_add_meta`: additive metacognitive bias

In addition, each parameter can be fitted in "duplex mode", such that separate values are fitted depending on the stimulus category (for type 1 parameters) or depending on the sign of the type 1 decision values (for type 2 parameters).

A more detailed guide to use the toolbox is provided in the following Jupyter notebook: [**Basic Usage**](https://github.com/m-guggenmos/remeta_v1/blob/main/demo/basic_usage.ipynb)
