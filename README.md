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
import remeta_v1 as remeta
x_stim, d_dec, c_conf = remeta.load_dataset('default')  # load example dataset
rem = remeta.ReMeta()
rem.fit(x_stim, d_dec, c_conf)
```
Output (for load_dataset):
```
Loading dataset 'default' which was generated as follows:
..Generative model:
    Type 2 noise type: noisy_report
    Type 2 noise distribution: beta
..Generative parameters:
    type1_noise: 0.5
    type1_bias: -0.1
    type2_noise: 7
    type2_criteria: [0.2, 0.2, 0.2, 0.2] = gaps | criteria = [0.2, 0.4, 0.6, 0.8]
..Characteristics:
    No. subjects: 1
    No. samples: 2000
    Type 1 performance: 86.5%
    Avg. confidence: 0.665
    M-Ratio: 0.862
    Criterion bias: 0
```
Output (for fit):
```
+++ Type 1 level +++
Initial guess (neg. LL: 692.75)
    [guess] type1_noise: 0.5
    [guess] type1_bias: 0
Performing local optimization
    [final] type1_noise: 0.503
    [final] type1_bias: -0.0821
Final neg. LL: 683.64
Total fitting time: 0.14 secs
+++ Type 2 level +++
Initial guess (neg. LL: 3902.60)
    [guess] type2_noise: 8.952
    [guess] type2_criteria_0: 0.2
    [guess] type2_criteria_1: 0.2 = gap | criterion = 0.4
    [guess] type2_criteria_2: 0.2 = gap | criterion = 0.6
    [guess] type2_criteria_3: 0.2 = gap | criterion = 0.8
Performing local optimization
    [final] type2_noise: 6.98
    [final] type2_criteria_0: 0.179
    [final] type2_criteria_1: 0.218 = gap | criterion = 0.398
    [final] type2_criteria_2: 0.203 = gap | criterion = 0.6
    [final] type2_criteria_3: 0.208 = gap | criterion = 0.809
    [extra] type2_criteria_absolute: [0.179, 0.398, 0.6, 0.809]
    [extra] type2_criteria_bias: 0.0107
Final neg. LL: 3312.87
Total fitting time: 12 secs
```

Since the dataset is based on simulation, we know the true parameters of the underlying generative model (see first output), which are quite close to the fitted parameters.

We can access the fitted parameters by invoking the `summary()` method on the `ReMeta` instance:

```python
# Access fitted parameters
result = rem.summary()
for k, v in result.model.params.items():
    print(f"{k}: {', '.join(f'{x:.3f}' for x in (v if hasattr(v, '__len__') else [v]))}")
```

Ouput:
```
type1_noise: 0.503
type1_bias: -0.082
type2_noise: 6.978
type2_criteria: 0.179, 0.218, 0.203, 0.208
```

By default, the model fits parameters for type 1 noise (`type1_noise`) and a type 1 bias (`type1_bias`), as well as metacognitive 'type 2' noise (`type2_noise`) and 4 confidence criteria (`type2_criteria`). Moreover, by default the model assumes that metacognitive noise occurs at the stage of the confidence report (setting `type2_noise_type='noisy_report'`) and that type 2 metacognitive noise can be described by a beta distribution (setting `type2_noise_dist='beta'`).

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
- `type2_criteria`: confidence criteria
- `type2_evidence_bias_mult`: optional multiplicative metacognitive bias

In addition, each type 1 parameter can be fitted in "duplex mode", such that separate values are fitted depending on the stimulus category.

A more detailed guide to use the toolbox is provided in the following Jupyter notebook: [**Basic Usage**](https://github.com/m-guggenmos/remeta_v1/blob/main/demo/basic_usage.ipynb)
