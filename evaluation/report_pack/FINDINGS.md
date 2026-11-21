# Findings: LLM serving optimization study (2 x H100 NVL)

Generated from `data/*.csv` by `scripts/report_pack/docs.py`; every number below is in a CSV named next to it. Claim ids (C1, C2, ...) refer to `CLAIMS.csv`. **Measured** = read directly off the data. **Interpretation** = our explanation of a measured pattern, with the evidence for it. **Open question** = seen in the data, not explained.

## Executive summary

- Five vLLM optimizations (FP8 weights, FP8 KV cache, EAGLE-3 speculative decoding, prefix caching, tuned batching) were measured in combination at 7 workload points on Llama-3.1-8B-Instruct: 384 runs in a factorial plan plus 126 focused re-runs with deep telemetry and 18 profiler traces.
- **We caught our own measurement error.** 71 of 384 factorial runs (18%) ran on a GPU another user's process was sharing; they read 0.60x of their true throughput. Correcting them cut the best reported speedup at 90% prefix reuse from 4.50x to 2.62x and at high concurrency from 2.64x to 1.67x.
- **Optimizations never cancel each other.** Across 50 combinations whose single-optimization speedups were all measured, none fell more than 5% below the product of its parts; 18 exceeded it by more than 5%, all of them with speculative decoding, up to +31%.
- **Two synergies have measured mechanisms:** prefix caching x speculative decoding grows with prompt reuse (1.01 at 0% reuse to 1.22 at 90%), and FP8 weights x speculative decoding (1.03-1.13 at the centre).
- **EAGLE-3 stops paying off beyond ~4k-token prompts:** 1.35x at 2048 tokens, 1.14x at 4096, 1.00x at 6144; acceptance of the 2nd draft token falls from 0.76 to 0.16 between 2k and 4k.
- **Accuracy:** FP8 weights alone and FP8 KV alone are statistically indistinguishable from BF16 on 1,319 GSM8K questions; together they lose 2.6 points (p = 0.03) - an accuracy interaction, not a throughput one.
- **The GPU, not the CPU, is the limit:** GPUs were busy 96%-99% of the profiled time and every run hit the 400 W power cap.

## F1. Hidden GPU interference, detected and corrected (methodology)

**Measured.** The Slurm node does not isolate GPUs (`CUDA_VISIBLE_DEVICES` unset, no GPU reservation); a pre-flight check the same morning, before the run started, found another user's vLLM server holding 85.6 GiB on GPU 1. From 11:24 UTC on 7 Nov until the end of the factorial run, every run on GPU 0 was slow:

| phase | runs | median throughput vs clean re-run | device memory used (MB) | SM clock (MHz) | power (W) |
|---|---|---|---|---|---|
| GPU 0 before 11:24 | 124 | 0.992 | 87498 | 1172 | 396 |
| GPU 0 from 11:24 | 71 | 0.596 | 91917 | 1476 | 387 |
| GPU 1 (whole run) | 189 | 1.006 | 87985 | 1171 | 395 |

- 71 runs in 24 cells at three workload points (high concurrency, 0% and 90% reuse) were affected; matched runs ran at 0.56-0.69x of the clean re-run, while GPU 1 at the same time stayed at 1.006.
- Signature of a co-tenant: +4.3 GB device memory beyond our server, higher SM clock and lower power (the GPU idles between time slices).
- **Correction:** the 8 affected cells that the investigation re-ran were replaced by the re-run; 16 cells have no clean value and are excluded (`cells_missing_after_cleaning.csv`). All tables in this pack use the cleaned dataset unless they say otherwise.
- **Lesson:** the slowdown was invisible within a cell (rep-to-rep CV 0.2-3.8%, consistent with ordinary noise); only re-running cells in a different session exposed it. Re-running a sample of cells is a cheap integrity check.

## F2. Best configuration per workload (corrected)

Value = throughput / Default throughput at the same workload, counted only if p99 TPOT <= 100 ms and p99 TTFT <= 5 s. Three accuracy policies are shown because the accuracy verdict changed with better data (F7).

| workload | Default tok/s | as first reported | A: original evals | B: paired evals | C: latency only |
|---|---|---|---|---|---|
| centre | 1774 | 1.82x | spec+prefix 1.82x | quant+spec+prefix 2.40x | quant+kvq+spec+prefix+batch 2.48x |
| short prompts | 4027 | 1.93x | spec+prefix 1.93x | quant+spec+prefix+batch 2.54x | quant+spec+prefix+batch 2.54x |
| long prompts | 495 | 1.50x | prefix+batch 1.50x | quant+spec+prefix+batch 1.82x | quant+kvq+prefix+batch 2.05x |
| low concurrency | 303 | 2.48x | spec+prefix 2.48x | quant+spec+prefix+batch 3.30x | quant+spec+prefix+batch 3.30x |
| high concurrency | 2162 | 2.64x | spec+prefix 1.67x | quant+spec+prefix+batch 2.12x | quant+spec+prefix+batch 2.12x |
| 0% reuse | 1776 | 1.38x | spec+prefix 1.39x | spec+prefix 1.39x | spec+prefix 1.39x |
| 90% reuse | 1775 | 4.50x | spec+prefix+batch 2.62x | quant+spec+prefix+batch 3.30x | quant+spec+prefix+batch 3.30x |

- **Policy A** reproduces the study's original rule (trimmed evals, 1-point gate on point estimates). **Policy B** uses the paired re-test: only settings with a statistically significant drop larger than the gate are excluded (FP8 weights + FP8 KV, and configs containing both). Its winners use FP8 weights + EAGLE-3 ('101'), a numeric setting the paired test did not cover: treat B as an upper bound pending that test. **Policy C** ignores accuracy.
- spec + prefix wins under policy A at 6 of 7 workloads; adding FP8 weights (policy B) raises the best speedup at every workload except 0% reuse.

## F3. Optimizations compound multiplicatively or better - they never cancel

**Measured.** For every combination we compare the observed speedup with the product of the speedups of its single optimizations measured at the same workload (`composition_per_config.csv`, basis = product of measured singles; 50 combinations at the centre, high concurrency, 0% and 90% reuse).

- multiplicative (independent): 32
- super-multiplicative (synergy): 18
- With speculative decoding: mean observed/predicted 1.111; without: 1.013.
- At the centre (all 32 configurations measured), the multiplicative prediction has median error 4.8% (max 21.6%); an additive prediction 1 + sum(s_i - 1) has median error 9.1% (max 37.4%). Speedups multiply; they do not add.

**Interpretation.** Each optimization removes a different cost: FP8 weights halve the bytes and double the math rate of every matrix multiply; FP8 KV halves attention-cache traffic; EAGLE-3 turns one decode step into ~3.4 output tokens; prefix caching deletes prompt tokens from prefill; tuned batching changes a limit that never binds here. Because they act on different terms of the per-token cost, none undoes another - and where two act on terms that grow when the other shrinks, they reinforce (F4, F5). See `CONFIG_NOTES.md` for every configuration.

## F4. Prefix caching x speculative decoding: synergy that grows with prompt reuse

| prefix reuse | prefix gain alone | prefix gain on top of spec | spec x prefix interaction |
|---|---|---|---|
| 0% | 1.01x | 1.02x | 1.011 |
| 10% | 1.02x | 1.03x | 1.015 |
| 25% | 1.11x | 1.12x | 1.014 |
| 50% | 1.20x | 1.35x | 1.122 |
| 90% | 1.52x | 1.85x | 1.217 |

**Measured mechanism (90% reuse).** Prefix caching shortens each request's decode phase by 27% without spec but by 45% with spec. With spec on, each engine step processes 955 tokens instead of 286.

**Interpretation (Amdahl's law).** vLLM processes the prompts of arriving requests in the same engine steps as the decode tokens of running ones. Speculative decoding cuts the cost per output token of the decode part, so prompt processing becomes a larger share of what remains - and that share is exactly what prefix caching removes. The more prompts are reused, the more there is to remove, so the synergy grows with reuse.

**Open question.** The synergy is flat up to 25% reuse and jumps between 25% and 50%; a finer reuse sweep would locate the threshold.

## F5. FP8 weights x speculative decoding

**Measured.** At the centre FP8 weights give 1.20x alone but 1.34x on top of EAGLE-3 (interaction 1.115; 1.03-1.13 across the other optimizations' settings). With spec, each engine step processes 928 tokens instead of 284.

**Interpretation.** Verifying 3 draft tokens per sequence makes every step a larger matrix multiply, moving it from memory-bound towards compute-bound; FP8 W8A8 doubles tensor-core math throughput, which helps compute-bound steps more than memory-bound ones.

## F6. EAGLE-3 breaks down on long prompts

| prompt tokens | spec speedup | acceptance | tokens/step | 1st draft | 2nd draft | 3rd draft |
|---|---|---|---|---|---|---|
| 2048 | 1.35x | 0.78 | 3.35 | 0.88 | 0.76 | 0.72 |
| 4096 | 1.14x | 0.37 | 2.12 | 0.88 | 0.16 | 0.07 |
| 6144 | 1.00x | 0.19 | 1.56 | 0.55 | 0.01 | 0.00 |
| 8192 | 1.09x | 0.10 | 1.31 | 0.31 | 0.00 | 0.00 |

**Interpretation (likely, not verified).** The draft head most likely saw only shorter contexts in training: later draft positions (which depend on the head's own earlier guesses) fail first, between 2k and 4k tokens; even the first draft position degrades past 4k. Speculative decoding should be switched off (or a long-context draft head used) for prompts beyond ~4k tokens.

**Open question.** Spec also cuts median time-to-first-token sharply (centre: 1046 -> 183 ms), which speculation does not obviously explain; it may come from shorter queueing when decode finishes faster.

## F7. Accuracy: an interaction between the two FP8 changes

Paired re-test: the same questions for Default and each setting; McNemar test on the questions whose answer changed.

| setting | task | n | Default | setting | change (pts) | 95% CI | p | verdict |
|---|---|---|---|---|---|---|---|---|
| quant | gsm8k | 1319 | 69.6 | 69.0 | -0.61 | [-2.6, +1.4] | 0.61 | not significant; CI too wide to confirm the gate |
| kvq | gsm8k | 1319 | 69.6 | 69.4 | -0.23 | [-2.3, +1.8] | 0.89 | not significant; CI too wide to confirm the gate |
| quant+kvq | gsm8k | 1319 | 69.6 | 67.0 | -2.58 | [-4.9, -0.3] | 0.03 | significant drop beyond gate |
| spec | gsm8k | 1319 | 69.6 | 69.2 | -0.38 | [-1.4, +0.6] | 0.56 | not significant; CI too wide to confirm the gate |
| quant | humaneval | 164 | 63.4 | 61.6 | -1.83 | [-5.8, +2.1] | 0.55 | not significant; CI too wide to confirm the gate |
| kvq | humaneval | 164 | 63.4 | 62.2 | -1.22 | [-5.7, +3.3] | 0.79 | not significant; CI too wide to confirm the gate |
| quant+kvq | humaneval | 164 | 63.4 | 62.2 | -1.22 | [-6.0, +3.6] | 0.80 | not significant; CI too wide to confirm the gate |
| spec | humaneval | 164 | 63.4 | 63.4 | +0.00 | [+0.0, +0.0] | 1.00 | within gate (CI excludes a larger drop) |
| quant | mmlu | 1687 | 65.7 | 65.9 | +0.18 | [-0.9, +1.2] | 0.83 | within gate (CI excludes a larger drop) |
| kvq | mmlu | 1687 | 65.7 | 65.3 | -0.47 | [-1.5, +0.6] | 0.45 | not significant; CI too wide to confirm the gate |
| quant+kvq | mmlu | 1687 | 65.7 | 65.3 | -0.41 | [-1.6, +0.7] | 0.55 | not significant; CI too wide to confirm the gate |
| spec | mmlu | 1687 | 65.7 | 65.6 | -0.12 | [-0.4, +0.2] | 0.73 | within gate (CI excludes a larger drop) |

- The original study excluded FP8 weights and FP8 KV on point estimates from trimmed evals (GSM8K 500 questions); the paired test shows neither alone is distinguishable from BF16, while the combination loses accuracy on GSM8K.
- EAGLE-3 changed no HumanEval result (pass/fail) and the result of 47 of 1,319 GSM8K questions (n.s.): numerically near-lossless, as speculative decoding should be.
- **Interpretation.** Each FP8 change adds a small rounding error; multi-step arithmetic (GSM8K) compounds both, so their errors add up past the gate while either alone stays within noise.

## F8. Which optimization matters where (clean main effects)

Throughput multiplier of each optimization averaged over the others (OLS on log throughput, +-1 coding; R^2 >= 0.998 at every point, residual noise 0.1%-0.7%).

| optimization | centre | short prompts | long prompts | low concurrency | high concurrency | 0% reuse | 90% reuse |
|---|---|---|---|---|---|---|---|
| quant | 1.27x | 1.27x | 1.23x | 1.34x | 1.27x | 1.22x | 1.25x |
| kvq | 1.04x | 1.00x | 1.13x | 1.00x | 1.07x | 1.04x | 1.05x |
| spec | 1.49x | 1.82x | 1.00x | 2.31x | 1.21x | 1.36x | 1.57x |
| prefix | 1.29x | 1.07x | 1.39x | 1.04x | 1.35x | 1.00x | 1.70x |
| batch | 1.00x | 1.00x | 1.02x | 1.00x | 1.01x | 1.00x | 1.00x |

- **spec** is the biggest lever when the GPU is under-used (low concurrency 2.31x) and useless on long prompts (1.00x, F6).
- **prefix** tracks prompt reuse (1.00x at 0%, 1.70x at 90%) and costs nothing when there is no reuse.
- **quant** is the most consistent: 1.22x-1.34x everywhere.
- **kvq** only matters where the KV cache is large: 1.13x on long prompts, <= 1.07x elsewhere.
- **tuned batching** never helps (1.00x-1.02x): the engine default (256 sequences) already exceeds every in-flight count, so the limit never binds - a null result with a clear cause, not a measurement failure.

## F9. The bottleneck is the GPU at its power cap, not the CPU

**Measured.** In 18 profiler traces the GPU executed kernels 96%-99% of the time; the top kernels are cuBLAS GEMMs (`nvjet_*`). NVML reported the 'SW power cap' throttle in all 126 investigation runs (mean power 394-400 W of 400 W). SM clocks fall as load rises: centre 1089 MHz, short prompts 1290 MHz, long prompts 888 MHz, low concurrency 1493 MHz, high concurrency 885 MHz, 0% reuse 1097 MHz, 90% reuse 1095 MHz (Default config).

- The factorial plan's first-order rules labelled every winner 'per-step overhead'; the traces contradict that - there is almost no idle GPU time to recover from CPU-side overhead.
- **Interpretation.** On this power-capped H100 NVL, throughput is bounded by GEMM work at reduced clocks; optimizations help to the extent they remove matrix-multiply work (FP8, spec, prefix), which matches F8.

## F10. The two-GPU method is sound: no contention between slots

The cell that looked like an outlier (FP8 KV + EAGLE-3 at 128 in flight) was re-run 10 times: 2690-2740 tok/s with the other GPU busy, 2666-2713 tok/s with it idle (cv < 1% each). Running two cells at once on two GPUs does not bias results; the original outlier was the F1 interference.

## Open questions and what was not measured

- 16 contaminated cells were never re-run (`cells_missing_after_cleaning.csv`).
- Paired accuracy for FP8 weights + spec, FP8 KV + spec and all three together.
- Why spec cuts time-to-first-token (F6); where between 25% and 50% reuse the spec x prefix synergy appears (F4).
- Workload is synthetic random-word text with fixed 256-token outputs; real chat or document traffic may change EAGLE-3 acceptance and prefix hit rates (see LIMITATIONS.md).
