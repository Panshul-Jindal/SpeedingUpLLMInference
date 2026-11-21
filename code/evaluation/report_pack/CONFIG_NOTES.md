# Configuration notes: why each combination performs as it does

Generated from `data/composition_per_config.csv`. **observed** = speedup over Default at that workload. **predicted** = what the configuration would give if its optimizations did not interact: the product of the single-optimization speedups measured at the same workload (or, where singles were not measured, a main-effects-only model fitted to that workload - marked *model*). **ratio** > 1.05 = synergy, < 0.95 = overlap. The mechanism line is read from that cell's own telemetry.

## centre (2048-token prompts, 32 in flight, 50% reuse)

Fastest: **quant+kvq+spec+prefix+batch** at 2.48x (latency gates not applied here; see FINDINGS F2). Synergies: 12; overlaps: 0.

| config | observed | predicted | ratio | basis | why (from telemetry) |
|---|---|---|---|---|---|
| default | 1.00x | - | - |  | Default: all off |
| batch | 1.00x | - | - |  | batch: limit 32 vs 28 running -> limit never binds |
| prefix | 1.21x | - | - |  | prefix: 37% of prompt tokens served from cache (prefill skipped) |
| spec | 1.36x | - | - |  | spec: EAGLE-3 accepts 78% of drafts -> 3.34 tokens per sequence per step |
| kvq | 1.04x | - | - |  | kvq: FP8 KV halves KV bytes; KV cache 7% full at most |
| quant | 1.20x | - | - |  | quant: FP8 weights halve the weight bytes read per step |
| prefix+batch | 1.21x | 1.21x | 0.997 | singles | prefix: 37% of prompt tokens served from cache (prefill skipped); batch: limit 32 vs 29 running -> limit never binds |
| spec+batch | 1.36x | 1.35x | 1.004 | singles | spec: EAGLE-3 accepts 78% of drafts -> 3.34 tokens per sequence per step; batch: limit 32 vs 31 running -> limit binds |
| spec+prefix | 1.82x | 1.64x | 1.110 | singles | spec: EAGLE-3 accepts 78% of drafts -> 3.35 tokens per sequence per step; prefix: 37% of prompt tokens served from cache (prefill skipped) |
| kvq+batch | 1.04x | 1.04x | 1.003 | singles | kvq: FP8 KV halves KV bytes; KV cache 7% full at most; batch: limit 32 vs 28 running -> limit never binds |
| kvq+prefix | 1.27x | 1.25x | 1.015 | singles | kvq: FP8 KV halves KV bytes; KV cache 6% full at most; prefix: 37% of prompt tokens served from cache (prefill skipped) |
| kvq+spec | 1.43x | 1.40x | 1.019 | singles | kvq: FP8 KV halves KV bytes; KV cache 7% full at most; spec: EAGLE-3 accepts 77% of drafts -> 3.32 tokens per sequence per step |
| quant+batch | 1.20x | 1.20x | 1.001 | singles | quant: FP8 weights halve the weight bytes read per step; batch: limit 32 vs 28 running -> limit never binds |
| quant+prefix | 1.51x | 1.45x | 1.038 | singles | quant: FP8 weights halve the weight bytes read per step; prefix: 37% of prompt tokens served from cache (prefill skipped) |
| quant+spec | 1.82x | 1.63x | 1.115 | singles | quant: FP8 weights halve the weight bytes read per step; spec: EAGLE-3 accepts 77% of drafts -> 3.32 tokens per sequence per step |
| quant+kvq | 1.23x | 1.24x | 0.990 | singles | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 6% full at most |
| spec+prefix+batch | 1.82x | 1.64x | 1.109 | singles | spec: EAGLE-3 accepts 78% of drafts -> 3.35 tokens per sequence per step; prefix: 37% of prompt tokens served from cache (prefill skipped); batch: limit 32 vs 31 running -> limit binds |
| kvq+prefix+batch | 1.28x | 1.25x | 1.018 | singles | kvq: FP8 KV halves KV bytes; KV cache 6% full at most; prefix: 37% of prompt tokens served from cache (prefill skipped); batch: limit 32 vs 29 running -> limit never binds |
| kvq+spec+batch | 1.40x | 1.40x | 0.997 | singles | kvq: FP8 KV halves KV bytes; KV cache 7% full at most; spec: EAGLE-3 accepts 78% of drafts -> 3.33 tokens per sequence per step; batch: limit 32 vs 31 running -> limit binds |
| kvq+spec+prefix | 1.91x | 1.70x | 1.126 | singles | kvq: FP8 KV halves KV bytes; KV cache 6% full at most; spec: EAGLE-3 accepts 78% of drafts -> 3.35 tokens per sequence per step; prefix: 37% of prompt tokens served from cache (prefill skipped) |
| quant+prefix+batch | 1.51x | 1.45x | 1.038 | singles | quant: FP8 weights halve the weight bytes read per step; prefix: 37% of prompt tokens served from cache (prefill skipped); batch: limit 32 vs 29 running -> limit never binds |
| quant+spec+batch | 1.82x | 1.63x | 1.116 | singles | quant: FP8 weights halve the weight bytes read per step; spec: EAGLE-3 accepts 77% of drafts -> 3.32 tokens per sequence per step; batch: limit 32 vs 31 running -> limit binds |
| quant+spec+prefix | 2.40x | 1.97x | 1.216 | singles | quant: FP8 weights halve the weight bytes read per step; spec: EAGLE-3 accepts 78% of drafts -> 3.33 tokens per sequence per step; prefix: 38% of prompt tokens served from cache (prefill skipped) |
| quant+kvq+batch | 1.24x | 1.24x | 0.995 | singles | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 6% full at most; batch: limit 32 vs 28 running -> limit never binds |
| quant+kvq+prefix | 1.58x | 1.51x | 1.049 | singles | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 5% full at most; prefix: 37% of prompt tokens served from cache (prefill skipped) |
| quant+kvq+spec | 1.85x | 1.69x | 1.100 | singles | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 6% full at most; spec: EAGLE-3 accepts 78% of drafts -> 3.34 tokens per sequence per step |
| kvq+spec+prefix+batch | 1.92x | 1.70x | 1.128 | singles | kvq: FP8 KV halves KV bytes; KV cache 6% full at most; spec: EAGLE-3 accepts 79% of drafts -> 3.36 tokens per sequence per step; prefix: 37% of prompt tokens served from cache (prefill skipped); batch: limit 32 vs 31 running -> limit binds |
| quant+spec+prefix+batch | 2.39x | 1.97x | 1.213 | singles | quant: FP8 weights halve the weight bytes read per step; spec: EAGLE-3 accepts 78% of drafts -> 3.33 tokens per sequence per step; prefix: 38% of prompt tokens served from cache (prefill skipped); batch: limit 32 vs 31 running -> limit binds |
| quant+kvq+prefix+batch | 1.58x | 1.51x | 1.047 | singles | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 5% full at most; prefix: 37% of prompt tokens served from cache (prefill skipped); batch: limit 32 vs 28 running -> limit never binds |
| quant+kvq+spec+batch | 1.88x | 1.69x | 1.116 | singles | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 8% full at most; spec: EAGLE-3 accepts 78% of drafts -> 3.34 tokens per sequence per step; batch: limit 32 vs 31 running -> limit binds |
| quant+kvq+spec+prefix | 2.46x | 2.04x | 1.203 | singles | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 5% full at most; spec: EAGLE-3 accepts 78% of drafts -> 3.34 tokens per sequence per step; prefix: 38% of prompt tokens served from cache (prefill skipped) |
| quant+kvq+spec+prefix+batch | 2.48x | 2.04x | 1.213 | singles | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 5% full at most; spec: EAGLE-3 accepts 78% of drafts -> 3.34 tokens per sequence per step; prefix: 38% of prompt tokens served from cache (prefill skipped); batch: limit 32 vs 31 running -> limit binds |

## short prompts (256)

Fastest: **quant+spec+prefix+batch** at 2.54x (latency gates not applied here; see FINDINGS F2). Synergies: 0; overlaps: 0.

| config | observed | predicted | ratio | basis | why (from telemetry) |
|---|---|---|---|---|---|
| default | 1.00x | - | - |  | Default: all off |
| prefix+batch | 1.05x | 1.07x | 0.981 | *model* | prefix: 38% of prompt tokens served from cache (prefill skipped); batch: limit 64 vs 31 running -> limit never binds |
| spec+batch | 1.79x | 1.83x | 0.982 | *model* | spec: EAGLE-3 accepts 72% of drafts -> 3.16 tokens per sequence per step; batch: limit 64 vs 31 running -> limit never binds |
| spec+prefix | 1.93x | 1.96x | 0.987 | *model* | spec: EAGLE-3 accepts 72% of drafts -> 3.16 tokens per sequence per step; prefix: 35% of prompt tokens served from cache (prefill skipped) |
| kvq+batch | 1.00x | 1.00x | 1.005 | *model* | kvq: FP8 KV halves KV bytes; KV cache 2% full at most; batch: limit 64 vs 32 running -> limit never binds |
| kvq+prefix | 1.06x | 1.07x | 0.990 | *model* | kvq: FP8 KV halves KV bytes; KV cache 1% full at most; prefix: 37% of prompt tokens served from cache (prefill skipped) |
| kvq+spec | 1.80x | 1.82x | 0.986 | *model* | kvq: FP8 KV halves KV bytes; KV cache 1% full at most; spec: EAGLE-3 accepts 74% of drafts -> 3.21 tokens per sequence per step |
| quant+batch | 1.25x | 1.27x | 0.986 | *model* | quant: FP8 weights halve the weight bytes read per step; batch: limit 64 vs 31 running -> limit never binds |
| quant+prefix | 1.34x | 1.36x | 0.986 | *model* | quant: FP8 weights halve the weight bytes read per step; prefix: 38% of prompt tokens served from cache (prefill skipped) |
| quant+spec | 2.30x | 2.32x | 0.993 | *model* | quant: FP8 weights halve the weight bytes read per step; spec: EAGLE-3 accepts 72% of drafts -> 3.16 tokens per sequence per step |
| quant+kvq | 1.28x | 1.27x | 1.004 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 1% full at most |
| kvq+spec+prefix+batch | 1.96x | 1.96x | 1.002 | *model* | kvq: FP8 KV halves KV bytes; KV cache 1% full at most; spec: EAGLE-3 accepts 74% of drafts -> 3.21 tokens per sequence per step; prefix: 35% of prompt tokens served from cache (prefill skipped); batch: limit 64 vs 31 running -> limit never binds |
| quant+spec+prefix+batch | 2.54x | 2.49x | 1.019 | *model* | quant: FP8 weights halve the weight bytes read per step; spec: EAGLE-3 accepts 72% of drafts -> 3.16 tokens per sequence per step; prefix: 34% of prompt tokens served from cache (prefill skipped); batch: limit 64 vs 31 running -> limit never binds |
| quant+kvq+prefix+batch | 1.34x | 1.36x | 0.981 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 1% full at most; prefix: 38% of prompt tokens served from cache (prefill skipped); batch: limit 64 vs 31 running -> limit never binds |
| quant+kvq+spec+batch | 2.27x | 2.32x | 0.978 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 1% full at most; spec: EAGLE-3 accepts 71% of drafts -> 3.13 tokens per sequence per step; batch: limit 64 vs 31 running -> limit never binds |
| quant+kvq+spec+prefix | 2.45x | 2.48x | 0.987 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 1% full at most; spec: EAGLE-3 accepts 71% of drafts -> 3.14 tokens per sequence per step; prefix: 34% of prompt tokens served from cache (prefill skipped) |

## long prompts (8192)

Fastest: **quant+kvq+prefix+batch** at 2.05x (latency gates not applied here; see FINDINGS F2). Synergies: 8; overlaps: 0.

| config | observed | predicted | ratio | basis | why (from telemetry) |
|---|---|---|---|---|---|
| default | 1.00x | - | - |  | Default: all off |
| spec (investigation) | 1.08x | - | - |  | spec: EAGLE-3 accepts 10% of drafts -> 1.31 tokens per sequence per step |
| prefix+batch | 1.50x | 1.42x | 1.059 | *model* | prefix: 39% of prompt tokens served from cache (prefill skipped); batch: limit 128 vs 31 running -> limit never binds |
| spec+batch | 1.05x | 1.02x | 1.034 | *model* | spec: EAGLE-3 accepts 10% of drafts -> 1.31 tokens per sequence per step; batch: limit 128 vs 32 running -> limit never binds |
| spec+prefix | 1.50x | 1.39x | 1.076 | *model* | spec: EAGLE-3 accepts 10% of drafts -> 1.30 tokens per sequence per step; prefix: 39% of prompt tokens served from cache (prefill skipped) |
| kvq+batch | 1.21x | 1.15x | 1.049 | *model* | kvq: FP8 KV halves KV bytes; KV cache 25% full at most; batch: limit 128 vs 31 running -> limit never binds |
| kvq+prefix | 1.66x | 1.57x | 1.057 | *model* | kvq: FP8 KV halves KV bytes; KV cache 19% full at most; prefix: 39% of prompt tokens served from cache (prefill skipped) |
| kvq+spec | 1.20x | 1.13x | 1.057 | *model* | kvq: FP8 KV halves KV bytes; KV cache 27% full at most; spec: EAGLE-3 accepts 14% of drafts -> 1.41 tokens per sequence per step |
| quant+batch | 1.37x | 1.25x | 1.092 | *model* | quant: FP8 weights halve the weight bytes read per step; batch: limit 128 vs 31 running -> limit never binds |
| quant+prefix | 1.80x | 1.71x | 1.051 | *model* | quant: FP8 weights halve the weight bytes read per step; prefix: 40% of prompt tokens served from cache (prefill skipped) |
| quant+spec | 1.29x | 1.23x | 1.043 | *model* | quant: FP8 weights halve the weight bytes read per step; spec: EAGLE-3 accepts 10% of drafts -> 1.30 tokens per sequence per step |
| quant+kvq | 1.51x | 1.40x | 1.083 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 23% full at most |
| kvq+spec+prefix+batch | 1.71x | 1.60x | 1.068 | *model* | kvq: FP8 KV halves KV bytes; KV cache 22% full at most; spec: EAGLE-3 accepts 13% of drafts -> 1.40 tokens per sequence per step; prefix: 39% of prompt tokens served from cache (prefill skipped); batch: limit 128 vs 32 running -> limit never binds |
| quant+spec+prefix+batch | 1.82x | 1.74x | 1.046 | *model* | quant: FP8 weights halve the weight bytes read per step; spec: EAGLE-3 accepts 10% of drafts -> 1.29 tokens per sequence per step; prefix: 39% of prompt tokens served from cache (prefill skipped); batch: limit 128 vs 32 running -> limit never binds |
| quant+kvq+prefix+batch | 2.05x | 1.98x | 1.036 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 18% full at most; prefix: 40% of prompt tokens served from cache (prefill skipped); batch: limit 128 vs 31 running -> limit never binds |
| quant+kvq+spec+batch | 1.48x | 1.42x | 1.043 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 24% full at most; spec: EAGLE-3 accepts 12% of drafts -> 1.36 tokens per sequence per step; batch: limit 128 vs 32 running -> limit never binds |
| quant+kvq+spec+prefix | 2.01x | 1.94x | 1.034 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 21% full at most; spec: EAGLE-3 accepts 12% of drafts -> 1.36 tokens per sequence per step; prefix: 39% of prompt tokens served from cache (prefill skipped) |

## low concurrency (2 in flight)

Fastest: **quant+spec+prefix+batch** at 3.30x (latency gates not applied here; see FINDINGS F2). Synergies: 0; overlaps: 0.

| config | observed | predicted | ratio | basis | why (from telemetry) |
|---|---|---|---|---|---|
| default | 1.00x | - | - |  | Default: all off |
| prefix+batch | 1.03x | 1.05x | 0.979 | *model* | prefix: 36% of prompt tokens served from cache (prefill skipped); batch: limit 16 vs 2 running -> limit never binds |
| spec+batch | 2.35x | 2.32x | 1.013 | *model* | spec: EAGLE-3 accepts 77% of drafts -> 3.31 tokens per sequence per step; batch: limit 16 vs 2 running -> limit never binds |
| spec+prefix | 2.48x | 2.41x | 1.029 | *model* | spec: EAGLE-3 accepts 77% of drafts -> 3.31 tokens per sequence per step; prefix: 33% of prompt tokens served from cache (prefill skipped) |
| kvq+batch | 1.03x | 1.00x | 1.024 | *model* | kvq: FP8 KV halves KV bytes; KV cache 0% full at most; batch: limit 16 vs 2 running -> limit never binds |
| kvq+prefix | 1.04x | 1.04x | 0.994 | *model* | kvq: FP8 KV halves KV bytes; KV cache 0% full at most; prefix: 36% of prompt tokens served from cache (prefill skipped) |
| kvq+spec | 2.36x | 2.31x | 1.022 | *model* | kvq: FP8 KV halves KV bytes; KV cache 0% full at most; spec: EAGLE-3 accepts 78% of drafts -> 3.34 tokens per sequence per step |
| quant+batch | 1.39x | 1.34x | 1.037 | *model* | quant: FP8 weights halve the weight bytes read per step; batch: limit 16 vs 2 running -> limit never binds |
| quant+prefix | 1.43x | 1.40x | 1.026 | *model* | quant: FP8 weights halve the weight bytes read per step; prefix: 36% of prompt tokens served from cache (prefill skipped) |
| quant+spec | 3.06x | 3.09x | 0.991 | *model* | quant: FP8 weights halve the weight bytes read per step; spec: EAGLE-3 accepts 77% of drafts -> 3.31 tokens per sequence per step |
| quant+kvq | 1.37x | 1.33x | 1.026 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 0% full at most |
| kvq+spec+prefix+batch | 2.50x | 2.42x | 1.033 | *model* | kvq: FP8 KV halves KV bytes; KV cache 0% full at most; spec: EAGLE-3 accepts 78% of drafts -> 3.34 tokens per sequence per step; prefix: 33% of prompt tokens served from cache (prefill skipped); batch: limit 16 vs 2 running -> limit never binds |
| quant+spec+prefix+batch | 3.30x | 3.24x | 1.020 | *model* | quant: FP8 weights halve the weight bytes read per step; spec: EAGLE-3 accepts 78% of drafts -> 3.34 tokens per sequence per step; prefix: 34% of prompt tokens served from cache (prefill skipped); batch: limit 16 vs 2 running -> limit never binds |
| quant+kvq+prefix+batch | 1.41x | 1.40x | 1.008 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 0% full at most; prefix: 36% of prompt tokens served from cache (prefill skipped); batch: limit 16 vs 2 running -> limit never binds |
| quant+kvq+spec+batch | 3.03x | 3.09x | 0.981 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 0% full at most; spec: EAGLE-3 accepts 78% of drafts -> 3.33 tokens per sequence per step; batch: limit 16 vs 2 running -> limit never binds |
| quant+kvq+spec+prefix | 3.24x | 3.22x | 1.006 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 0% full at most; spec: EAGLE-3 accepts 78% of drafts -> 3.34 tokens per sequence per step; prefix: 34% of prompt tokens served from cache (prefill skipped) |

## high concurrency (128 in flight)

Fastest: **quant+spec+prefix+batch** at 2.12x (latency gates not applied here; see FINDINGS F2). Synergies: 1; overlaps: 0.

| config | observed | predicted | ratio | basis | why (from telemetry) |
|---|---|---|---|---|---|
| default (re-run) | 1.00x | - | - |  | Default: all off |
| batch (investigation) | 1.01x | - | - |  | batch: limit 128 vs 120 running -> limit never binds |
| prefix (investigation) | 1.34x | - | - |  | prefix: 36% of prompt tokens served from cache (prefill skipped) |
| spec (investigation) | 1.20x | - | - |  | spec: EAGLE-3 accepts 80% of drafts -> 3.41 tokens per sequence per step |
| kvq (investigation) | 1.08x | - | - |  | kvq: FP8 KV halves KV bytes; KV cache 27% full at most |
| quant (investigation) | 1.25x | - | - |  | quant: FP8 weights halve the weight bytes read per step |
| prefix+batch | 1.34x | 1.34x | 0.994 | singles | prefix: 36% of prompt tokens served from cache (prefill skipped); batch: limit 128 vs 118 running -> limit never binds |
| spec+prefix | 1.67x | 1.60x | 1.043 | singles | spec: EAGLE-3 accepts 80% of drafts -> 3.39 tokens per sequence per step; prefix: 36% of prompt tokens served from cache (prefill skipped) |
| kvq+batch | 1.12x | 1.09x | 1.033 | singles | kvq: FP8 KV halves KV bytes; KV cache 27% full at most; batch: limit 128 vs 119 running -> limit never binds |
| kvq+prefix (re-run) | 1.42x | 1.45x | 0.980 | singles | kvq: FP8 KV halves KV bytes; KV cache 20% full at most; prefix: 36% of prompt tokens served from cache (prefill skipped) |
| kvq+spec (re-run) | 1.25x | 1.29x | 0.967 | singles | kvq: FP8 KV halves KV bytes; KV cache 28% full at most; spec: EAGLE-3 accepts 81% of drafts -> 3.44 tokens per sequence per step |
| quant+batch | 1.26x | 1.26x | 0.999 | singles | quant: FP8 weights halve the weight bytes read per step; batch: limit 128 vs 120 running -> limit never binds |
| quant+prefix (re-run) | 1.64x | 1.68x | 0.979 | singles | quant: FP8 weights halve the weight bytes read per step; prefix: 36% of prompt tokens served from cache (prefill skipped) |
| quant+kvq | 1.38x | 1.36x | 1.018 | singles | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 25% full at most |
| spec+prefix+batch (investigation) | 1.66x | 1.61x | 1.034 | singles | spec: EAGLE-3 accepts 80% of drafts -> 3.39 tokens per sequence per step; prefix: 36% of prompt tokens served from cache (prefill skipped); batch: limit 128 vs 125 running -> limit never binds |
| kvq+spec+prefix+batch | 1.75x | 1.74x | 1.004 | singles | kvq: FP8 KV halves KV bytes; KV cache 20% full at most; spec: EAGLE-3 accepts 81% of drafts -> 3.43 tokens per sequence per step; prefix: 36% of prompt tokens served from cache (prefill skipped); batch: limit 128 vs 125 running -> limit never binds |
| quant+spec+prefix+batch | 2.12x | 2.02x | 1.054 | singles | quant: FP8 weights halve the weight bytes read per step; spec: EAGLE-3 accepts 79% of drafts -> 3.36 tokens per sequence per step; prefix: 36% of prompt tokens served from cache (prefill skipped); batch: limit 128 vs 125 running -> limit never binds |
| quant+kvq+spec+batch | 1.64x | 1.63x | 1.006 | singles | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 25% full at most; spec: EAGLE-3 accepts 79% of drafts -> 3.38 tokens per sequence per step; batch: limit 128 vs 125 running -> limit never binds |

## no prefix reuse (0%)

Fastest: **spec+prefix** at 1.39x (latency gates not applied here; see FINDINGS F2). Synergies: 0; overlaps: 0.

| config | observed | predicted | ratio | basis | why (from telemetry) |
|---|---|---|---|---|---|
| default | 1.00x | - | - |  | Default: all off |
| prefix (investigation) | 1.01x | - | - |  | prefix: 0% of prompt tokens served from cache (prefill skipped) |
| spec (investigation) | 1.37x | - | - |  | spec: EAGLE-3 accepts 80% of drafts -> 3.40 tokens per sequence per step |
| prefix+batch | 1.00x | 1.00x | 1.001 | *model* | prefix: 0% of prompt tokens served from cache (prefill skipped); batch: limit 64 vs 29 running -> limit never binds |
| spec+batch | 1.38x | 1.38x | 1.004 | *model* | spec: EAGLE-3 accepts 80% of drafts -> 3.40 tokens per sequence per step; batch: limit 64 vs 31 running -> limit never binds |
| spec+prefix (re-run) | 1.39x | 1.37x | 1.014 | singles | spec: EAGLE-3 accepts 80% of drafts -> 3.40 tokens per sequence per step; prefix: 0% of prompt tokens served from cache (prefill skipped) |
| kvq+batch | 1.05x | 1.05x | 0.997 | *model* | kvq: FP8 KV halves KV bytes; KV cache 7% full at most; batch: limit 64 vs 27 running -> limit never binds |
| kvq+prefix | 1.05x | 1.05x | 0.993 | *model* | kvq: FP8 KV halves KV bytes; KV cache 7% full at most; prefix: 0% of prompt tokens served from cache (prefill skipped) |
| quant+prefix | 1.19x | 1.21x | 0.991 | *model* | quant: FP8 weights halve the weight bytes read per step; prefix: 0% of prompt tokens served from cache (prefill skipped) |
| quant+kvq | 1.27x | 1.27x | 1.003 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 6% full at most |
| spec+prefix+batch (investigation) | 1.36x | 1.38x | 0.990 | *model* | spec: EAGLE-3 accepts 80% of drafts -> 3.40 tokens per sequence per step; prefix: 0% of prompt tokens served from cache (prefill skipped); batch: limit 64 vs 31 running -> limit never binds |
| quant+kvq+prefix+batch | 1.27x | 1.27x | 1.003 | *model* | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 6% full at most; prefix: 0% of prompt tokens served from cache (prefill skipped); batch: limit 64 vs 28 running -> limit never binds |

## 90% prefix reuse

Fastest: **quant+spec+prefix+batch** at 3.30x (latency gates not applied here; see FINDINGS F2). Synergies: 5; overlaps: 0.

| config | observed | predicted | ratio | basis | why (from telemetry) |
|---|---|---|---|---|---|
| default (re-run) | 1.00x | - | - |  | Default: all off |
| batch (investigation) | 1.00x | - | - |  | batch: limit 128 vs 28 running -> limit never binds |
| prefix (investigation) | 1.52x | - | - |  | prefix: 68% of prompt tokens served from cache (prefill skipped) |
| spec (investigation) | 1.40x | - | - |  | spec: EAGLE-3 accepts 82% of drafts -> 3.46 tokens per sequence per step |
| kvq (investigation) | 1.04x | - | - |  | kvq: FP8 KV halves KV bytes; KV cache 7% full at most |
| quant (investigation) | 1.19x | - | - |  | quant: FP8 weights halve the weight bytes read per step |
| prefix+batch | 1.54x | 1.52x | 1.018 | singles | prefix: 68% of prompt tokens served from cache (prefill skipped); batch: limit 128 vs 30 running -> limit never binds |
| spec+prefix | 2.62x | 2.12x | 1.235 | singles | spec: EAGLE-3 accepts 82% of drafts -> 3.45 tokens per sequence per step; prefix: 67% of prompt tokens served from cache (prefill skipped) |
| kvq+batch | 1.05x | 1.04x | 1.012 | singles | kvq: FP8 KV halves KV bytes; KV cache 7% full at most; batch: limit 128 vs 29 running -> limit never binds |
| kvq+prefix (re-run) | 1.60x | 1.58x | 1.014 | singles | kvq: FP8 KV halves KV bytes; KV cache 4% full at most; prefix: 68% of prompt tokens served from cache (prefill skipped) |
| quant+batch | 1.19x | 1.19x | 1.003 | singles | quant: FP8 weights halve the weight bytes read per step; batch: limit 128 vs 28 running -> limit never binds |
| quant+prefix (re-run) | 1.86x | 1.80x | 1.033 | singles | quant: FP8 weights halve the weight bytes read per step; prefix: 67% of prompt tokens served from cache (prefill skipped) |
| quant+kvq | 1.27x | 1.23x | 1.030 | singles | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 6% full at most |
| spec+prefix+batch (investigation) | 2.62x | 2.12x | 1.239 | singles | spec: EAGLE-3 accepts 81% of drafts -> 3.44 tokens per sequence per step; prefix: 67% of prompt tokens served from cache (prefill skipped); batch: limit 128 vs 31 running -> limit never binds |
| kvq+spec+prefix+batch | 2.69x | 2.20x | 1.222 | singles | kvq: FP8 KV halves KV bytes; KV cache 4% full at most; spec: EAGLE-3 accepts 81% of drafts -> 3.42 tokens per sequence per step; prefix: 67% of prompt tokens served from cache (prefill skipped); batch: limit 128 vs 31 running -> limit never binds |
| quant+spec+prefix+batch | 3.30x | 2.52x | 1.311 | singles | quant: FP8 weights halve the weight bytes read per step; spec: EAGLE-3 accepts 78% of drafts -> 3.33 tokens per sequence per step; prefix: 67% of prompt tokens served from cache (prefill skipped); batch: limit 128 vs 31 running -> limit never binds |
| quant+kvq+spec+batch | 1.86x | 1.72x | 1.083 | singles | quant: FP8 weights halve the weight bytes read per step; kvq: FP8 KV halves KV bytes; KV cache 6% full at most; spec: EAGLE-3 accepts 77% of drafts -> 3.30 tokens per sequence per step; batch: limit 128 vs 31 running -> limit never binds |

