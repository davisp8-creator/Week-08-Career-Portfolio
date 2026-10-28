# Results (pe_ablation, tag `cpu`)

Metric: test MAE in kW (lower is better), 5 seeds per arm. Reference arm: `sinusoidal`. 'Exceeds noise' = Welch p < 0.05 after Holm correction across 3 comparisons.

## Per-arm results

| arm | n_seeds | n_params | mae_mean | mae_sd | mae_ci95_halfwidth | mae_min | mae_max | rmse_mean | train_s_mean | p50_ms_window | p95_ms_window | p50_round_sd | peak_gpu_mb | rss_delta_mb_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5 | 68824 | 21.385 | 0.813 | 1.009 | 20.190 | 22.291 | 29.067 | 171.028 | 0.324 | 0.393 | 0.006 | nan | 246.302 |
| sinusoidal | 5 | 68824 | 8.428 | 0.129 | 0.160 | 8.285 | 8.571 | 13.008 | 169.610 | 0.316 | 0.386 | 0.008 | nan | -166.977 |
| learned | 5 | 79576 | 8.580 | 0.183 | 0.227 | 8.455 | 8.896 | 13.247 | 168.943 | 0.310 | 0.376 | 0.010 | nan | -39.323 |
| rope | 5 | 68824 | 8.429 | 0.098 | 0.121 | 8.324 | 8.589 | 13.006 | 187.522 | 0.425 | 0.497 | 0.004 | nan | 50.563 |

## Each arm vs reference

| arm | reference | diff_mae_kw | diff_pct_of_ref | welch_ci_lo | welch_ci_hi | welch_p | paired_p | effect_over_seed_sd | welch_p_holm | exceeds_noise |
|---|---|---|---|---|---|---|---|---|---|---|
| none | sinusoidal | 12.9570 | 153.7351 | 11.9539 | 13.9602 | 0.0000 | 0.0000 | 22.2616 | 0.0000 | True |
| learned | sinusoidal | 0.1522 | 1.8053 | -0.0830 | 0.3873 | 0.1707 | 0.2066 | 0.9628 | 0.3413 | False |
| rope | sinusoidal | 0.0013 | 0.0151 | -0.1673 | 0.1699 | 0.9864 | 0.9897 | 0.0112 | 0.9864 | False |

No-learning references: same hours yesterday MAE = 12.37 kW; same hours last week = 11.84 kW.

## Verdict

- none vs sinusoidal: EXCEEDS run-to-run noise (+12.957 kW, +153.7%, none worse; Holm-adjusted p=7.12e-06; effect = 22.3x the seed SD).
- learned vs sinusoidal: NOT distinguishable from noise (+0.152 kW, +1.8%; Welch 95% CI [-0.083, +0.387] includes 0; Holm-adjusted p=0.341).
- rope vs sinusoidal: NOT distinguishable from noise (+0.001 kW, +0.0%; Welch 95% CI [-0.167, +0.170] includes 0; Holm-adjusted p=0.986).

Caveat: with 5 seeds per arm the test has low power; 'not distinguishable' means 'not shown', not 'equal'. All seeds share one synthetic dataset, so the seed spread excludes data-sampling variance.