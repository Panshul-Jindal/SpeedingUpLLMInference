# Data dictionary

All CSVs are in `data/`. Throughput in output tokens/s; times in ms unless the column says `_s`. Workload points: centre = 2048-token prompts, 32 requests in flight, 50% prefix reuse; each other point changes one of those (context 256/8192, concurrency 2/128, reuse 0%/90%).

## Common columns

- `config_bits`: 7 characters, optimization 1 first: quant, kvq, spec, prefix, batch, chunk, par. '1011100' = quant+spec+prefix+batch. chunk and par are always 0 in this study. Read as text (leading zeros matter).
- `output_tok_s`: Output tokens per second over the 120 s measurement window (higher is better).
- `speedup`: output_tok_s / Default output_tok_s at the same point.
- `value`: speedup if all gates pass, else 1.0 (study definition).
- `tpot_p99_ms`: 99th-percentile time per output token after the first (gate: <= 100 ms).
- `ttft_p99_ms`: 99th-percentile time to first token (gate: <= 5000 ms).
- `spec_accept_rate`: Accepted / proposed EAGLE-3 draft tokens.
- `tokens_per_step`: Output tokens per sequence per engine step (1 + accepted/drafts).
- `prefix_hit_rate`: Share of prompt tokens served from the prefix cache.
- `deep_tokens_per_step`: Tokens (prompt + decode) processed per engine step (vllm:iteration_tokens_total).
- `deep_decode_s / deep_prefill_s`: Mean time a request spends in its decode / prefill phase.
- `deep_cpu_*, deep_read_gbps, deep_tflops`: NOT usable in this run (collector bug / counters not enabled; fixed afterwards). Ignore.
- `contaminated`: True for fast-plan runs on GPU 0 from 11:24 UTC 7 Nov (co-tenant process).
- `source`: fast = clean factorial run; re-run = replaces a contaminated cell; investigation = extra cell from the investigation.
- `ratio_observed_to_independent`: observed / predicted speedup; > 1.05 synergy, < 0.95 overlap.
- `independent_basis`: How the no-interaction prediction was made: product of measured singles (strong) or main-effects model (weak).
- `gpu_busy_frac`: Share of the profiled span during which a GPU kernel was running.

## Files

### binding_check.csv

Bottleneck evidence (other GPU idle): steps/s, utilisation, clocks, power, profiler GPU-busy fraction, top kernel.

Rows: 7. Columns: `point_id`, `config`, `tok_s`, `running_mean`, `deep_steps_per_s`, `deep_step_ms`, `deep_sm_util_mean`, `deep_mem_util_mean`, `deep_sm_clock_mean_mhz`, `deep_power_mean_w`, `gpu_busy_frac`, `top_kernel`, `power_cap_w`, `throttle_reason`

### cells_combined_clean.csv

One row per (workload point, configuration) from the cleaned dataset: geometric-mean throughput, speedup, gates, telemetry means.

Rows: 128. Columns: `env`, `point_id`, `config_bits`, `config_names`, `n_reps_ok`, `n_reps`, `default_tok_s`, `accuracy_gate`, `max_accuracy_drop`, `status`, `output_tok_s`, `speedup`, `p99_tpot_ms`, `p99_ttft_ms`, `failed_gates`, `value`, `error`, `default_fails_gate`, `default_failed_gates`, `source`, `tok_s_cv`, `spec_accept_rate`, `tokens_per_step`, `prefix_hit_rate`, `running_mean`, `kv_usage_max`, `preemptions`, `max_num_seqs`, `sm_clock_mhz`, `power_w`, `tpot_p50_ms`, `ttft_p50_ms`, `latency_gates_pass`, `n_opts`, `missing`

### cells_missing_after_cleaning.csv

Contaminated cells with no clean value (excluded from all analysis).

Rows: 16. Columns: `point_id`, `config_bits`, `config`, `reason`

### clean_config_x_regime.csv

Value for every configuration at every point (cleaned data).

Rows: 33. Columns: `env`, `config_bits`, `config_names`, `center`, `context_low`, `context_high`, `concurrency_low`, `concurrency_high`, `reuse_low`, `reuse_high`

### clean_final_configs.csv

Harness-standard final configuration per point (original accuracy policy) with binding-constraint label.

Rows: 7. Columns: `env`, `point_id`, `final_config`, `final_names`, `value`, `output_tok_s`, `default_tok_s`, `default_fails_gate`, `default_failed_gates`, `binding_constraint`, `kv_usage_max`, `preemptions`, `running_max`, `waiting_mean`, `max_num_seqs`, `bw_gbps`, `bw_frac`, `flops_tflops`, `flops_frac`, `notes`

### clean_interactions.csv

Two-way interaction ratios from the harness analysis on the cleaned data (labelled measured or model).

Rows: 70. Columns: `env`, `point_id`, `a`, `b`, `pair`, `ratio`, `label`, `se_log`

### clean_main_effects.csv

Main-effect multipliers with standard errors per point (harness analysis on the cleaned dataset).

Rows: 35. Columns: `env`, `point_id`, `optimization`, `name`, `multiplier`, `se`, `beta`, `beta_se`, `status`

### clean_model_fit.csv

R^2, residual and noise per point for the clean OLS fits.

Rows: 7. Columns: `env`, `point_id`, `n_obs`, `n_configs`, `n_params`, `n_interaction_terms`, `r2`, `residual_sd_log`, `df_resid`, `noise_sd_log`, `status`

### clean_no_benefit.csv

Optimizations with < 2% main effect per point, with a telemetry reason.

Rows: 11. Columns: `env`, `point_id`, `optimization`, `name`, `multiplier`, `gain_pct`, `reason`

### clean_shapley.csv

Shapley shares of each final configuration's log gain.

Rows: 15. Columns: `env`, `point_id`, `final_config`, `optimization`, `name`, `phi_log`, `share`, `total_log_gain`, `exact`, `label`, `status`

### composition_per_config.csv

Observed vs no-interaction predicted speedup for every configuration, with a mechanism line from telemetry.

Rows: 128. Columns: `point_id`, `config_bits`, `config`, `n_opts`, `source`, `speedup_observed`, `speedup_independent`, `independent_basis`, `ratio_observed_to_independent`, `speedup_main_effects_model`, `speedup_product_of_singles`, `speedup_additive_singles`, `mechanism`, `verdict`

### contamination_by_phase.csv

GPU 0 before / after 11:24 UTC vs GPU 1: runs, ratio to re-run, device memory, clock, power.

Rows: 3. Columns: `phase`, `runs`, `device_mem_used_mb`, `power_w`, `sm_clock_mhz`, `first`, `last`, `median_ratio_to_rerun`

### contamination_matched_runs.csv

Fast-plan runs whose cell was re-run cleanly: ratio fast/re-run over time, by GPU.

Rows: 63. Columns: `written_at`, `gpu`, `point_id`, `config_bits`, `rep`, `output_tok_s`, `rerun_tok_s`, `ratio_fast_to_rerun`, `contaminated`, `gpu_mem_used_max_mb`, `gpu_power_mean_w`, `gpu_sm_clock_mean_mhz`

### fast_plan_as_first_reported.csv

Best Value per workload as the contaminated factorial plan first reported it.

Rows: 7. Columns: `point_id`, `best_value_as_first_reported`

### final_configs_by_policy.csv

Best configuration per workload under accuracy policies A/B/C (latency gates always on).

Rows: 21. Columns: `point_id`, `policy`, `best_config`, `best_names`, `speedup`, `tok_s`, `default_tok_s`, `source`, `accuracy_status_B`

### interactions_measured_2x2.csv

Interaction ratio I = T_AB*T_0/(T_A*T_B) wherever all four cells of a 2x2 square were measured (1 = independent).

Rows: 119. Columns: `point_id`, `with`, `pair`, `gain_a`, `gain_b`, `gain_ab`, `interaction`, `source`

### investigation_cells.csv

Investigation per-cell table (all points incl. derived ones, shared and exclusive runs).

Rows: 60. Columns: `point_id`, `config_bits`, `config`, `exclusive`, `tok_s`, `tok_s_cv`, `reps_ok`, `tpot_p50_ms`, `tpot_p99_ms`, `ttft_p50_ms`, `ttft_p99_ms`, `spec_accept_rate`, `tokens_per_step`, `prefix_hit_rate`, `running_mean`, `kv_usage_max`, `preemptions`, `gpu_util_mean`, `deep_sm_util_mean`, `deep_mem_util_mean`, `deep_sm_clock_mean_mhz`, `deep_sm_clock_min_mhz`, `deep_power_mean_w`, `deep_temp_max_c`, `deep_throttle_reasons`, `deep_steps_per_s`, `deep_step_ms`, `deep_tokens_per_step`, `deep_read_gbps`, `deep_write_gbps`, `deep_tflops`, `deep_prefill_s`, `deep_decode_s`, `deep_itl_s`, `deep_queue_s`, `deep_cached_prompt_frac`, `deep_cpu_engine_core_pct`, `deep_cpu_engine_core_max_pct`, `deep_cpu_api_server_pct`, `deep_cpu_api_server_max_pct`, `deep_cpu_host_pct`, `deep_spec_accept_by_pos`, `deep_read_frac_peak`, `deep_tflops_frac_peak`, `fast_tok_s`, `vs_fast`

### outlier_rerun.csv

The 10 re-runs of FP8 KV + EAGLE-3 at high concurrency (other GPU busy vs idle).

Rows: 10. Columns: `exclusive`, `rep`, `gpu`, `output_tok_s`, `tpot_p99_ms`, `spec_accept_rate`

### prefix_vs_reuse.csv

Prefix-caching gain with and without spec at 0/10/25/50/90% prompt reuse.

Rows: 5. Columns: `reuse`, `default`, `prefix`, `spec`, `spec+prefix`, `source`, `prefix_gain`, `prefix_gain_with_spec`, `spec_gain`, `spec_prefix_interaction`

### profiler_passes.csv

All 18 vLLM torch-profiler passes: GPU-busy fraction over 64 engine steps, top kernel and CPU op.

Rows: 18. Columns: `point_id`, `config_bits`, `exclusive`, `config`, `status`, `gpu_busy_frac`, `span_ms`, `cpu_step_ms_mean`, `top_kernel`, `top_cpu_op`, `error`

### quality_paired_vs_default.csv

Paired accuracy re-test (GSM8K 1319, HumanEval 164, MMLU 8 subjects = 1687): difference, 95% CI, McNemar p, verdict.

Rows: 12. Columns: `setting`, `names`, `task`, `n`, `base_acc`, `acc`, `diff_pts`, `ci95_lo`, `ci95_hi`, `lost`, `gained`, `mcnemar_p`, `verdict`

### quality_trimmed_fast_plan.csv

Accuracy of the 8 numeric settings in the factorial plan (trimmed: GSM8K 500, HumanEval 164, MMLU 4 subjects).

Rows: 8. Columns: `numeric_bits`, `names`, `gsm8k_pts`, `humaneval_pts`, `mmlu_pts`, `agreement_with_default`, `samples`

### runs_combined_clean.csv

The cleaned dataset used everywhere: clean factorial runs + re-runs of contaminated cells + extra investigation cells at the 7 points (`source`).

Rows: 363. Columns: `written_at`, `run_key`, `kind`, `profile`, `env`, `config_bits`, `config_names`, `point_id`, `rep`, `context`, `concurrency`, `reuse`, `n_active`, `overrides`, `trace_hash`, `seed`, `dry_run`, `vllm_version`, `torch_version`, `cuda_version`, `driver_version`, `lm_eval_version`, `harness_git`, `layout`, `gpu_ids`, `flags`, `flags_key`, `tuned_values`, `used_placeholder`, `engine_info`, `startup_s`, `status`, `window_s`, `n_requests_window`, `n_requests_total`, `n_errors`, `n_wrong_output_len`, `output_tok_s`, `req_s`, `ttft_p50_ms`, `ttft_p99_ms`, `ttft_mean_ms`, `tpot_p50_ms`, `tpot_p99_ms`, `tpot_mean_ms`, `e2e_p50_ms`, `e2e_p99_ms`, `e2e_mean_ms`, `trace_wrapped`, `kv_usage_mean`, `kv_usage_max`, `running_mean`, `running_max`, `waiting_mean`, `waiting_max`, `preemptions`, `prefix_hits`, `prefix_queries`, `queue_time_mean_s`, `prompt_tokens`, `generation_tokens`, `prefix_hit_rate`, `spec_accept_rate`, `tokens_per_step`, `gpu_util_mean`, `gpu_mem_used_max_mb`, `gpu_power_mean_w`, `gpu_sm_clock_mean_mhz`, `telemetry_missing`, `prefix_cache_reset`, `max_num_seqs_effective`, `duration_s`, `spec_draft_tokens`, `spec_accepted_tokens`, `spec_drafts`, `gpu`, `source`, `investigations`, `tuned_point`, `exclusive`, `deep_sm_util_mean`, `deep_mem_util_mean`, `deep_sm_clock_mean_mhz`, `deep_sm_clock_min_mhz`, `deep_power_mean_w`, `deep_temp_max_c`, `deep_throttle_reasons`, `deep_steps_per_s`, `deep_step_ms`, `deep_tokens_per_step`, `deep_read_gbps`, `deep_write_gbps`, `deep_tflops`, `deep_prefill_s`, `deep_decode_s`, `deep_itl_s`, `deep_queue_s`, `deep_cached_prompt_frac`, `deep_cpu_engine_core_pct`, `deep_cpu_engine_core_max_pct`, `deep_cpu_api_server_pct`, `deep_cpu_api_server_max_pct`, `deep_cpu_host_pct`, `deep_spec_accept_by_pos`, `profile_files`, `profile_output_tok_s`, `replaced_from`

### runs_fast_plan_all.csv

Every factorial-plan run (384) as measured, with the `contaminated` flag (GPU 0 after 11:24 UTC).

Rows: 384. Columns: `written_at`, `run_key`, `kind`, `profile`, `env`, `config_bits`, `config_names`, `point_id`, `rep`, `context`, `concurrency`, `reuse`, `n_active`, `overrides`, `trace_hash`, `seed`, `dry_run`, `vllm_version`, `torch_version`, `cuda_version`, `driver_version`, `lm_eval_version`, `harness_git`, `layout`, `gpu_ids`, `flags`, `flags_key`, `tuned_values`, `used_placeholder`, `engine_info`, `startup_s`, `status`, `window_s`, `n_requests_window`, `n_requests_total`, `n_errors`, `n_wrong_output_len`, `output_tok_s`, `req_s`, `ttft_p50_ms`, `ttft_p99_ms`, `ttft_mean_ms`, `tpot_p50_ms`, `tpot_p99_ms`, `tpot_mean_ms`, `e2e_p50_ms`, `e2e_p99_ms`, `e2e_mean_ms`, `trace_wrapped`, `kv_usage_mean`, `kv_usage_max`, `running_mean`, `running_max`, `waiting_mean`, `waiting_max`, `preemptions`, `prefix_hits`, `prefix_queries`, `queue_time_mean_s`, `prompt_tokens`, `generation_tokens`, `prefix_hit_rate`, `spec_accept_rate`, `tokens_per_step`, `gpu_util_mean`, `gpu_mem_used_max_mb`, `gpu_power_mean_w`, `gpu_sm_clock_mean_mhz`, `telemetry_missing`, `prefix_cache_reset`, `max_num_seqs_effective`, `duration_s`, `spec_draft_tokens`, `spec_accepted_tokens`, `spec_drafts`, `gpu`, `contaminated`

### runs_investigation.csv

All 126 investigation runs with deep telemetry (`deep_*` columns), incl. derived points such as center@reuse=0.1.

Rows: 126. Columns: `written_at`, `run_key`, `kind`, `profile`, `env`, `config_bits`, `config_names`, `point_id`, `rep`, `context`, `concurrency`, `reuse`, `n_active`, `overrides`, `trace_hash`, `seed`, `dry_run`, `vllm_version`, `torch_version`, `cuda_version`, `driver_version`, `lm_eval_version`, `harness_git`, `investigations`, `tuned_point`, `exclusive`, `layout`, `gpu_ids`, `flags`, `flags_key`, `tuned_values`, `used_placeholder`, `engine_info`, `startup_s`, `status`, `window_s`, `n_requests_window`, `n_requests_total`, `n_errors`, `n_wrong_output_len`, `output_tok_s`, `req_s`, `ttft_p50_ms`, `ttft_p99_ms`, `ttft_mean_ms`, `tpot_p50_ms`, `tpot_p99_ms`, `tpot_mean_ms`, `e2e_p50_ms`, `e2e_p99_ms`, `e2e_mean_ms`, `trace_wrapped`, `kv_usage_mean`, `kv_usage_max`, `running_mean`, `running_max`, `waiting_mean`, `waiting_max`, `preemptions`, `prefix_hits`, `prefix_queries`, `queue_time_mean_s`, `prompt_tokens`, `generation_tokens`, `prefix_hit_rate`, `spec_accept_rate`, `tokens_per_step`, `gpu_util_mean`, `gpu_mem_used_max_mb`, `gpu_power_mean_w`, `gpu_sm_clock_mean_mhz`, `deep_sm_util_mean`, `deep_mem_util_mean`, `deep_sm_clock_mean_mhz`, `deep_sm_clock_min_mhz`, `deep_power_mean_w`, `deep_temp_max_c`, `deep_throttle_reasons`, `deep_steps_per_s`, `deep_step_ms`, `deep_tokens_per_step`, `deep_read_gbps`, `deep_write_gbps`, `deep_tflops`, `deep_prefill_s`, `deep_decode_s`, `deep_itl_s`, `deep_queue_s`, `deep_cached_prompt_frac`, `deep_cpu_engine_core_pct`, `deep_cpu_engine_core_max_pct`, `deep_cpu_api_server_pct`, `deep_cpu_api_server_max_pct`, `deep_cpu_host_pct`, `telemetry_missing`, `prefix_cache_reset`, `max_num_seqs_effective`, `duration_s`, `spec_draft_tokens`, `spec_accepted_tokens`, `spec_drafts`, `deep_spec_accept_by_pos`, `profile_files`, `profile_output_tok_s`, `gpu`

### spec_vs_context.csv

EAGLE-3 speedup and acceptance by draft position at 2048-8192-token prompts.

Rows: 4. Columns: `context_tokens`, `default_tok_s`, `spec_tok_s`, `spec_speedup`, `accept_rate`, `tokens_per_step`, `accept_pos1`, `accept_pos2`, `accept_pos3`, `default_tpot_p50_ms`, `spec_tpot_p50_ms`, `default_ttft_p50_ms`, `spec_ttft_p50_ms`

