from remeta_v1 import remeta
cfg = remeta.Configuration()
cfg.type2_noise_type = 'noisy_report'
cfg.enable_type2_param_evidence_bias_mult = 1
cfg.enable_type2_param_evidence_bias_add = 1

params_true = dict(
    type1_noise=0.5,
    type1_bias=-0.1,
    type2_noise=0.1,
    type2_evidence_bias_mult=1.3,
    type2_evidence_bias_add=0.1
)

data = remeta.simu_data(nsubjects=1, nsamples=5000, params=params_true, squeeze=True, x_stim_stepsize=0.25, cfg=cfg)

cfg.true_params = params_true
rem = remeta.ReMeta(cfg=cfg)
rem.fit(data.x_stim, data.d_dec, data.c_conf)