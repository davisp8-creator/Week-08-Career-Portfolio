# Results (pe_ablation, tag `t4gpu`)

Metric: test MAE in kW (lower is better), 5 seeds per arm. Reference arm: `sinusoidal`. 'Exceeds noise' = Welch p < 0.05 after Holm correction across 3 comparisons.

## Per-arm results

| arm | n_seeds | n_params | mae_mean | mae_sd | mae_ci95_halfwidth | mae_min | mae_max | rmse_mean | train_s_mean | p50_ms_window | p95_ms_window | p50_round_sd | peak_gpu_mb | rss_delta_mb_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 5 | 68824 | 21.060 | 0.801 | 0.995 | 20.174 | 22.257 | 28.780 | 22.288 | 0.067 | 0.068 | 0.000 | 527.992 | 69.616 |
| sinusoidal | 5 | 68824 | 8.449 | 0.132 | 0.164 | 8.230 | 8.567 | 13.051 | 22.338 | 0.067 | 0.069 | 0.000 | 528.074 | 0.003 |
| learned | 5 | 79576 | 8.542 | 0.134 | 0.166 | 8.406 | 8.728 | 13.198 | 22.280 | 0.067 | 0.069 | 0.000 | 528.197 | 0.002 |
| rope | 5 | 68824 | 8.473 | 0.114 | 0.142 | 8.340 | 8.628 | 13.063 | 26.747 | 0.078 | 0.079 | 0.000 | 570.014 | 0.072 |

## Each arm vs reference

| arm | reference | diff_mae_kw | diff_pct_of_ref | welch_ci_lo | welch_ci_hi | welch_p | paired_p | effect_over_seed_sd | welch_p_holm | exceeds_noise |
|---|---|---|---|---|---|---|---|---|---|---|
| none | sinusoidal | 12.6116 | 149.2744 | 11.6237 | 13.5995 | 0.0000 | 0.0000 | 21.9703 | 0.0000 | True |
| learned | sinusoidal | 0.0930 | 1.1011 | -0.1008 | 0.2868 | 0.3005 | 0.2790 | 0.7001 | 0.6010 | False |
| rope | sinusoidal | 0.0239 | 0.2835 | -0.1567 | 0.2046 | 0.7670 | 0.8025 | 0.1940 | 0.7670 | False |

No-learning references: same hours yesterday MAE = 12.37 kW; same hours last week = 11.84 kW.

## Verdict

- none vs sinusoidal: EXCEEDS run-to-run noise (+12.612 kW, +149.3%, none worse; Holm-adjusted p=7.23e-06; effect = 22.0x the seed SD).
- learned vs sinusoidal: NOT distinguishable from noise (+0.093 kW, +1.1%; Welch 95% CI [-0.101, +0.287] includes 0; Holm-adjusted p=0.601).
- rope vs sinusoidal: NOT distinguishable from noise (+0.024 kW, +0.3%; Welch 95% CI [-0.157, +0.205] includes 0; Holm-adjusted p=0.767).

Caveat: with 5 seeds per arm the test has low power; 'not distinguishable' means 'not shown', not 'equal'. All seeds share one synthetic dataset, so the seed spread excludes data-sampling variance.